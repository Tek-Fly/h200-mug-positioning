"""
Rule storage and version control.

Manages rule persistence in MongoDB with version tracking.
"""

# Standard library imports
from datetime import datetime
from typing import Any, Dict, List, Optional

# Third-party imports
import structlog
from bson import ObjectId

# First-party imports
from src.database.mongodb import get_mongodb_client

# Local imports
from .models import Rule, RuleType

# Initialize logger
logger = structlog.get_logger(__name__)


class RuleStorage:
    """
    Manages rule storage in MongoDB with version control.

    Features:
    - CRUD operations for rules
    - Version tracking and history
    - Rule searching and filtering
    - Batch operations
    - Audit logging
    """

    def __init__(self, database_name: str = "h200_mug_positioning"):
        """
        Initialize rule storage.

        Args:
            database_name: MongoDB database name
        """
        self.client = None
        self.db = None
        self.rules_collection = None
        self.history_collection = None
        self.database_name = database_name

    async def initialize(self):
        """Initialize database connection and collections."""
        if not self.client:
            self.client = await get_mongodb_client()
            self.db = self.client[self.database_name]
            self.rules_collection = self.db.rules
            self.history_collection = self.db.rule_history

            # Create indexes
            await self._create_indexes()

            logger.info("RuleStorage initialized", database=self.database_name)

    async def _create_indexes(self):
        """Create database indexes for performance."""
        # Rules collection indexes
        await self.rules_collection.create_index("name", unique=True)
        await self.rules_collection.create_index("type")
        await self.rules_collection.create_index("priority")
        await self.rules_collection.create_index("enabled")
        await self.rules_collection.create_index("created_at")
        await self.rules_collection.create_index(
            [("name", "text"), ("description", "text")]
        )

        # History collection indexes
        await self.history_collection.create_index("rule_id")
        await self.history_collection.create_index("version")
        await self.history_collection.create_index("created_at")
        await self.history_collection.create_index([("rule_id", 1), ("version", -1)])

    async def create_rule(self, rule: Rule, user_id: str = "system") -> str:
        """
        Create a new rule.

        Args:
            rule: Rule to create
            user_id: ID of user creating the rule

        Returns:
            Created rule ID
        """
        await self.initialize()

        # Set creation metadata
        rule.created_by = user_id
        rule.created_at = datetime.utcnow()
        rule.updated_at = rule.created_at
        rule.version = 1

        # Convert to dict and insert
        rule_dict = rule.to_dict()

        try:
            result = await self.rules_collection.insert_one(rule_dict)
            rule_id = str(result.inserted_id)

            # Create initial history entry
            await self._create_history_entry(rule_id, rule_dict, "created", user_id)

            logger.info(
                "Rule created", rule_id=rule_id, rule_name=rule.name, user_id=user_id
            )

            return rule_id

        except Exception as e:
            logger.error("Failed to create rule", rule_name=rule.name, error=str(e))
            raise

    async def get_rule(self, rule_id: str) -> Optional[Rule]:
        """
        Get a rule by ID.

        Args:
            rule_id: Rule ID

        Returns:
            Rule object or None if not found
        """
        await self.initialize()

        try:
            rule_dict = await self.rules_collection.find_one({"_id": ObjectId(rule_id)})

            if rule_dict:
                return Rule.from_dict(rule_dict)

            return None

        except Exception as e:
            logger.error("Failed to get rule", rule_id=rule_id, error=str(e))
            return None

    async def get_rule_by_name(self, name: str) -> Optional[Rule]:
        """
        Get a rule by name.

        Args:
            name: Rule name

        Returns:
            Rule object or None if not found
        """
        await self.initialize()

        rule_dict = await self.rules_collection.find_one({"name": name})

        if rule_dict:
            return Rule.from_dict(rule_dict)

        return None

    async def update_rule(
        self, rule_id: str, updates: Dict[str, Any], user_id: str = "system"
    ) -> bool:
        """
        Update a rule.

        Args:
            rule_id: Rule ID
            updates: Fields to update
            user_id: ID of user updating the rule

        Returns:
            True if successful
        """
        await self.initialize()

        try:
            # Get current rule for history
            current = await self.rules_collection.find_one({"_id": ObjectId(rule_id)})

            if not current:
                return False

            # Increment version and update timestamp
            updates["version"] = current["version"] + 1
            updates["updated_at"] = datetime.utcnow()

            # Update rule
            result = await self.rules_collection.update_one(
                {"_id": ObjectId(rule_id)}, {"$set": updates}
            )

            if result.modified_count > 0:
                # Create history entry
                await self._create_history_entry(
                    rule_id, current, "updated", user_id, updates
                )

                logger.info(
                    "Rule updated",
                    rule_id=rule_id,
                    version=updates["version"],
                    user_id=user_id,
                )

                return True

            return False

        except Exception as e:
            logger.error("Failed to update rule", rule_id=rule_id, error=str(e))
            return False

    async def delete_rule(
        self, rule_id: str, user_id: str = "system", soft_delete: bool = True
    ) -> bool:
        """
        Delete a rule.

        Args:
            rule_id: Rule ID
            user_id: ID of user deleting the rule
            soft_delete: If True, disable rule instead of deleting

        Returns:
            True if successful
        """
        await self.initialize()

        try:
            if soft_delete:
                # Soft delete - just disable the rule
                return await self.update_rule(rule_id, {"enabled": False}, user_id)
            else:
                # Hard delete
                current = await self.rules_collection.find_one(
                    {"_id": ObjectId(rule_id)}
                )

                if not current:
                    return False

                # Delete rule
                result = await self.rules_collection.delete_one(
                    {"_id": ObjectId(rule_id)}
                )

                if result.deleted_count > 0:
                    # Create final history entry
                    await self._create_history_entry(
                        rule_id, current, "deleted", user_id
                    )

                    logger.info("Rule deleted", rule_id=rule_id, user_id=user_id)

                    return True

            return False

        except Exception as e:
            logger.error("Failed to delete rule", rule_id=rule_id, error=str(e))
            return False

    async def list_rules(
        self,
        filter_dict: Optional[Dict[str, Any]] = None,
        sort_by: str = "priority",
        sort_order: int = -1,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Rule]:
        """
        List rules with filtering and pagination.

        Args:
            filter_dict: MongoDB filter
            sort_by: Field to sort by
            sort_order: 1 for ascending, -1 for descending
            limit: Maximum number of rules to return
            skip: Number of rules to skip

        Returns:
            List of rules
        """
        await self.initialize()

        filter_dict = filter_dict or {}

        cursor = self.rules_collection.find(filter_dict)
        cursor = cursor.sort(sort_by, sort_order)
        cursor = cursor.skip(skip).limit(limit)

        rules = []
        async for rule_dict in cursor:
            rules.append(Rule.from_dict(rule_dict))

        return rules

    async def search_rules(
        self,
        query: str,
        rule_type: Optional[RuleType] = None,
        enabled_only: bool = True,
    ) -> List[Rule]:
        """
        Search rules by text.

        Args:
            query: Search query
            rule_type: Filter by rule type
            enabled_only: Only return enabled rules

        Returns:
            List of matching rules
        """
        await self.initialize()

        filter_dict = {"$text": {"$search": query}}

        if rule_type:
            filter_dict["type"] = rule_type.value

        if enabled_only:
            filter_dict["enabled"] = True

        return await self.list_rules(filter_dict)

    async def get_rules_by_type(
        self, rule_type: RuleType, enabled_only: bool = True
    ) -> List[Rule]:
        """
        Get all rules of a specific type.

        Args:
            rule_type: Type of rules to retrieve
            enabled_only: Only return enabled rules

        Returns:
            List of rules
        """
        filter_dict = {"type": rule_type.value}

        if enabled_only:
            filter_dict["enabled"] = True

        return await self.list_rules(filter_dict)

    async def get_active_rules(self) -> List[Rule]:
        """
        Get all active (enabled) rules sorted by priority.

        Returns:
            List of active rules
        """
        return await self.list_rules(
            {"enabled": True}, sort_by="priority", sort_order=-1
        )

    async def batch_create_rules(
        self, rules: List[Rule], user_id: str = "system"
    ) -> List[str]:
        """
        Create multiple rules in batch.

        Args:
            rules: List of rules to create
            user_id: ID of user creating the rules

        Returns:
            List of created rule IDs
        """
        rule_ids = []

        for rule in rules:
            try:
                rule_id = await self.create_rule(rule, user_id)
                rule_ids.append(rule_id)
            except Exception as e:
                logger.error(
                    "Failed to create rule in batch", rule_name=rule.name, error=str(e)
                )

        return rule_ids

    async def get_rule_history(
        self, rule_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get version history for a rule.

        Args:
            rule_id: Rule ID
            limit: Maximum number of history entries

        Returns:
            List of history entries
        """
        await self.initialize()

        cursor = (
            self.history_collection.find({"rule_id": rule_id})
            .sort("version", -1)
            .limit(limit)
        )

        history = []
        async for entry in cursor:
            entry["_id"] = str(entry["_id"])
            history.append(entry)

        return history

    async def restore_rule_version(
        self, rule_id: str, version: int, user_id: str = "system"
    ) -> bool:
        """
        Restore a rule to a previous version.

        Args:
            rule_id: Rule ID
            version: Version to restore
            user_id: ID of user restoring the rule

        Returns:
            True if successful
        """
        await self.initialize()

        # Get historical version
        history_entry = await self.history_collection.find_one(
            {"rule_id": rule_id, "version": version}
        )

        if not history_entry:
            return False

        # Restore rule data
        rule_data = history_entry["rule_data"]
        rule_data.pop("_id", None)  # Remove ID to avoid conflicts

        # Update with restore metadata
        rule_data["restored_from_version"] = version
        rule_data["restored_by"] = user_id
        rule_data["restored_at"] = datetime.utcnow()

        return await self.update_rule(rule_id, rule_data, user_id)

    async def _create_history_entry(
        self,
        rule_id: str,
        rule_data: Dict[str, Any],
        action: str,
        user_id: str,
        changes: Optional[Dict[str, Any]] = None,
    ):
        """Create a history entry for rule changes."""
        history_entry = {
            "rule_id": rule_id,
            "version": rule_data.get("version", 1),
            "action": action,
            "rule_data": rule_data.copy(),
            "changes": changes,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
        }

        await self.history_collection.insert_one(history_entry)

    async def export_rules(
        self, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Export rules as JSON-serializable dictionaries.

        Args:
            filter_dict: Optional filter

        Returns:
            List of rule dictionaries
        """
        rules = await self.list_rules(filter_dict)

        return [
            {
                "id": rule.id,
                "name": rule.name,
                "description": rule.description,
                "natural_language": rule.natural_language,
                "type": rule.type.value,
                "conditions": [c.to_dict() for c in rule.conditions],
                "actions": [a.to_dict() for a in rule.actions],
                "priority": rule.priority,
                "enabled": rule.enabled,
                "version": rule.version,
                "created_at": rule.created_at.isoformat(),
                "updated_at": rule.updated_at.isoformat(),
                "created_by": rule.created_by,
                "metadata": rule.metadata,
            }
            for rule in rules
        ]

    async def import_rules(
        self,
        rule_dicts: List[Dict[str, Any]],
        user_id: str = "system",
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Import rules from dictionaries.

        Args:
            rule_dicts: List of rule dictionaries
            user_id: ID of user importing rules
            overwrite: Whether to overwrite existing rules

        Returns:
            Import summary
        """
        summary = {"imported": 0, "skipped": 0, "errors": 0, "details": []}

        for rule_dict in rule_dicts:
            try:
                # Check if rule exists
                existing = await self.get_rule_by_name(rule_dict["name"])

                if existing and not overwrite:
                    summary["skipped"] += 1
                    summary["details"].append(
                        {
                            "name": rule_dict["name"],
                            "status": "skipped",
                            "reason": "already exists",
                        }
                    )
                    continue

                # Create Rule object
                rule = Rule(
                    name=rule_dict["name"],
                    description=rule_dict["description"],
                    natural_language=rule_dict["natural_language"],
                    type=RuleType(rule_dict["type"]),
                    priority=rule_dict["priority"],
                    enabled=rule_dict.get("enabled", True),
                    metadata=rule_dict.get("metadata", {}),
                )

                # Import conditions and actions would be parsed here
                # (simplified for brevity)

                if existing and overwrite:
                    await self.update_rule(existing.id, rule.to_dict(), user_id)
                else:
                    await self.create_rule(rule, user_id)

                summary["imported"] += 1
                summary["details"].append(
                    {"name": rule_dict["name"], "status": "imported"}
                )

            except Exception as e:
                summary["errors"] += 1
                summary["details"].append(
                    {
                        "name": rule_dict.get("name", "unknown"),
                        "status": "error",
                        "error": str(e),
                    }
                )

        return summary
