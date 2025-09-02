"""Unit tests for rules engine."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.core.rules.engine import RulesEngine
from src.core.rules.models import Rule, RuleType, Condition, Action
from src.core.rules.parser import NaturalLanguageParser
from src.core.rules.executor import RuleExecutor
from src.core.rules.storage import RuleStorage
from tests.base import AsyncBaseTest


@pytest.mark.unit
class TestRulesEngine(AsyncBaseTest):
    """Test cases for RulesEngine."""
    
    async def setup_method_async(self):
        """Setup rules engine test."""
        await super().setup_method_async()
        
        self.mock_storage = AsyncMock(spec=RuleStorage)
        self.mock_parser = Mock(spec=NaturalLanguageParser)
        self.mock_executor = AsyncMock(spec=RuleExecutor)
        
        self.engine = RulesEngine(
            storage=self.mock_storage,
            parser=self.mock_parser,
            executor=self.mock_executor
        )
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test rules engine initialization."""
        engine = RulesEngine()
        await engine.initialize()
        
        assert engine.initialized
        assert engine.storage is not None
        assert engine.parser is not None
        assert engine.executor is not None
    
    @pytest.mark.asyncio
    async def test_create_rule_from_text_success(self):
        """Test creating rule from natural language."""
        rule_text = "If mug confidence is greater than 0.8, then highlight position"
        
        # Mock parser response
        mock_rule = Rule(
            id="rule_123",
            name="High confidence highlighting",
            rule_type=RuleType.CONDITIONAL,
            conditions=[
                Condition(
                    field="mug_confidence",
                    operator="greater_than",
                    value=0.8
                )
            ],
            actions=[
                Action(
                    type="highlight_position",
                    parameters={"style": "bold"}
                )
            ],
            created_at=datetime.now(),
            active=True
        )
        self.mock_parser.parse.return_value = mock_rule
        
        # Mock storage
        self.mock_storage.save_rule.return_value = "rule_123"
        
        result = await self.engine.create_rule_from_text(rule_text)
        
        assert result == "rule_123"
        self.mock_parser.parse.assert_called_once_with(rule_text)
        self.mock_storage.save_rule.assert_called_once_with(mock_rule)
    
    @pytest.mark.asyncio
    async def test_create_rule_from_text_parse_failure(self):
        """Test rule creation with parsing failure."""
        rule_text = "Invalid rule text that cannot be parsed"
        
        self.mock_parser.parse.side_effect = ValueError("Cannot parse rule")
        
        with pytest.raises(ValueError, match="Cannot parse rule"):
            await self.engine.create_rule_from_text(rule_text)
    
    @pytest.mark.asyncio
    async def test_get_rule_success(self):
        """Test getting existing rule."""
        mock_rule = Rule(
            id="rule_123",
            name="Test Rule",
            rule_type=RuleType.CONDITIONAL,
            conditions=[],
            actions=[],
            created_at=datetime.now(),
            active=True
        )
        self.mock_storage.get_rule.return_value = mock_rule
        
        result = await self.engine.get_rule("rule_123")
        
        assert result == mock_rule
        self.mock_storage.get_rule.assert_called_once_with("rule_123")
    
    @pytest.mark.asyncio
    async def test_get_rule_not_found(self):
        """Test getting non-existent rule."""
        self.mock_storage.get_rule.return_value = None
        
        result = await self.engine.get_rule("nonexistent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_list_rules(self):
        """Test listing rules."""
        mock_rules = [
            Rule(id="rule_1", name="Rule 1", rule_type=RuleType.CONDITIONAL, 
                 conditions=[], actions=[], created_at=datetime.now(), active=True),
            Rule(id="rule_2", name="Rule 2", rule_type=RuleType.CONDITIONAL,
                 conditions=[], actions=[], created_at=datetime.now(), active=False)
        ]
        self.mock_storage.list_rules.return_value = mock_rules
        
        result = await self.engine.list_rules()
        
        assert len(result) == 2
        assert result[0].id == "rule_1"
        assert result[1].id == "rule_2"
    
    @pytest.mark.asyncio
    async def test_list_rules_filtered(self):
        """Test listing rules with filters."""
        mock_rules = [
            Rule(id="rule_1", name="Rule 1", rule_type=RuleType.CONDITIONAL,
                 conditions=[], actions=[], created_at=datetime.now(), active=True)
        ]
        self.mock_storage.list_rules.return_value = mock_rules
        
        result = await self.engine.list_rules(active_only=True)
        
        assert len(result) == 1
        self.mock_storage.list_rules.assert_called_once_with(active_only=True)
    
    @pytest.mark.asyncio
    async def test_update_rule_success(self):
        """Test updating existing rule."""
        rule_id = "rule_123"
        updates = {"name": "Updated Rule Name", "active": False}
        
        self.mock_storage.update_rule.return_value = True
        
        result = await self.engine.update_rule(rule_id, updates)
        
        assert result is True
        self.mock_storage.update_rule.assert_called_once_with(rule_id, updates)
    
    @pytest.mark.asyncio
    async def test_delete_rule_success(self):
        """Test deleting rule."""
        rule_id = "rule_123"
        
        self.mock_storage.delete_rule.return_value = True
        
        result = await self.engine.delete_rule(rule_id)
        
        assert result is True
        self.mock_storage.delete_rule.assert_called_once_with(rule_id)
    
    @pytest.mark.asyncio
    async def test_evaluate_rules_success(self):
        """Test evaluating rules against analysis data."""
        analysis_data = {
            "mug_confidence": 0.9,
            "position_x": 150,
            "position_y": 200
        }
        
        mock_results = [
            {"rule_id": "rule_1", "executed": True, "actions": ["highlight"]},
            {"rule_id": "rule_2", "executed": False, "reason": "condition_not_met"}
        ]
        self.mock_executor.evaluate_rules.return_value = mock_results
        
        result = await self.engine.evaluate_rules(analysis_data)
        
        assert len(result) == 2
        assert result[0]["executed"] is True
        self.mock_executor.evaluate_rules.assert_called_once_with(analysis_data)
    
    @pytest.mark.asyncio
    async def test_validate_rule_success(self):
        """Test rule validation."""
        mock_rule = Rule(
            id="rule_123",
            name="Valid Rule",
            rule_type=RuleType.CONDITIONAL,
            conditions=[
                Condition(field="confidence", operator="greater_than", value=0.5)
            ],
            actions=[Action(type="alert", parameters={})],
            created_at=datetime.now(),
            active=True
        )
        
        # Rule validation should pass
        result = self.engine.validate_rule(mock_rule)
        
        assert result is True
    
    def test_validate_rule_invalid_conditions(self):
        """Test validation with invalid conditions."""
        mock_rule = Rule(
            id="rule_123",
            name="Invalid Rule",
            rule_type=RuleType.CONDITIONAL,
            conditions=[],  # Empty conditions
            actions=[Action(type="alert", parameters={})],
            created_at=datetime.now(),
            active=True
        )
        
        with pytest.raises(ValueError, match="Rule must have at least one condition"):
            self.engine.validate_rule(mock_rule)
    
    def test_validate_rule_invalid_actions(self):
        """Test validation with invalid actions."""
        mock_rule = Rule(
            id="rule_123",
            name="Invalid Rule",
            rule_type=RuleType.CONDITIONAL,
            conditions=[
                Condition(field="confidence", operator="greater_than", value=0.5)
            ],
            actions=[],  # Empty actions
            created_at=datetime.now(),
            active=True
        )
        
        with pytest.raises(ValueError, match="Rule must have at least one action"):
            self.engine.validate_rule(mock_rule)


@pytest.mark.unit
class TestNaturalLanguageParser:
    """Test cases for NaturalLanguageParser."""
    
    def setup_method(self):
        """Setup parser test."""
        self.parser = NaturalLanguageParser()
    
    def test_parse_simple_condition(self):
        """Test parsing simple condition."""
        text = "If confidence is greater than 0.8, then send alert"
        
        rule = self.parser.parse(text)
        
        assert rule.name is not None
        assert rule.rule_type == RuleType.CONDITIONAL
        assert len(rule.conditions) == 1
        assert rule.conditions[0].field == "confidence"
        assert rule.conditions[0].operator == "greater_than"
        assert rule.conditions[0].value == 0.8
        assert len(rule.actions) == 1
        assert rule.actions[0].type == "send_alert"
    
    def test_parse_multiple_conditions(self):
        """Test parsing multiple conditions."""
        text = "If confidence > 0.8 and position_x < 100, then highlight and notify"
        
        rule = self.parser.parse(text)
        
        assert len(rule.conditions) == 2
        assert rule.conditions[0].field == "confidence"
        assert rule.conditions[1].field == "position_x"
        assert len(rule.actions) == 2
    
    def test_parse_complex_rule(self):
        """Test parsing complex rule with parameters."""
        text = "If mug confidence is greater than 0.9 and position is in region 'kitchen', then highlight with color 'red' and send notification to 'admin@example.com'"
        
        rule = self.parser.parse(text)
        
        assert len(rule.conditions) == 2
        assert rule.conditions[0].field == "mug_confidence"
        assert rule.conditions[1].field == "position_region"
        assert len(rule.actions) == 2
        assert rule.actions[0].parameters.get("color") == "red"
        assert rule.actions[1].parameters.get("recipient") == "admin@example.com"
    
    def test_parse_invalid_syntax(self):
        """Test parsing invalid syntax."""
        text = "This is not a valid rule"
        
        with pytest.raises(ValueError, match="Cannot parse rule"):
            self.parser.parse(text)
    
    def test_parse_unsupported_operator(self):
        """Test parsing unsupported operator."""
        text = "If confidence is approximately 0.8, then alert"
        
        with pytest.raises(ValueError, match="Unsupported operator"):
            self.parser.parse(text)


@pytest.mark.unit  
class TestRuleExecutor(AsyncBaseTest):
    """Test cases for RuleExecutor."""
    
    async def setup_method_async(self):
        """Setup executor test."""
        await super().setup_method_async()
        
        self.mock_storage = AsyncMock(spec=RuleStorage)
        self.executor = RuleExecutor(storage=self.mock_storage)
    
    @pytest.mark.asyncio
    async def test_evaluate_rules_success(self):
        """Test successful rule evaluation."""
        analysis_data = {
            "confidence": 0.9,
            "position_x": 150,
            "mug_detected": True
        }
        
        # Mock rules
        rules = [
            Rule(
                id="rule_1",
                name="High Confidence Rule",
                rule_type=RuleType.CONDITIONAL,
                conditions=[
                    Condition(field="confidence", operator="greater_than", value=0.8)
                ],
                actions=[Action(type="highlight", parameters={"color": "green"})],
                created_at=datetime.now(),
                active=True
            ),
            Rule(
                id="rule_2", 
                name="Position Rule",
                rule_type=RuleType.CONDITIONAL,
                conditions=[
                    Condition(field="position_x", operator="less_than", value=100)
                ],
                actions=[Action(type="alert", parameters={})],
                created_at=datetime.now(),
                active=True
            )
        ]
        
        self.mock_storage.list_rules.return_value = rules
        
        results = await self.executor.evaluate_rules(analysis_data)
        
        assert len(results) == 2
        # First rule should execute (confidence 0.9 > 0.8)
        assert results[0]["executed"] is True
        assert results[0]["rule_id"] == "rule_1"
        # Second rule should not execute (position_x 150 not < 100)
        assert results[1]["executed"] is False
        assert results[1]["rule_id"] == "rule_2"
    
    @pytest.mark.asyncio
    async def test_evaluate_single_rule_true_condition(self):
        """Test evaluating single rule with true condition."""
        analysis_data = {"confidence": 0.9}
        
        rule = Rule(
            id="rule_1",
            name="Test Rule",
            rule_type=RuleType.CONDITIONAL,
            conditions=[
                Condition(field="confidence", operator="greater_than", value=0.8)
            ],
            actions=[Action(type="highlight", parameters={})],
            created_at=datetime.now(),
            active=True
        )
        
        with patch.object(self.executor, '_execute_actions', return_value=["highlight"]) as mock_execute:
            result = await self.executor._evaluate_single_rule(rule, analysis_data)
            
            assert result["executed"] is True
            assert result["actions"] == ["highlight"]
            mock_execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_evaluate_single_rule_false_condition(self):
        """Test evaluating single rule with false condition."""
        analysis_data = {"confidence": 0.5}
        
        rule = Rule(
            id="rule_1",
            name="Test Rule",
            rule_type=RuleType.CONDITIONAL,
            conditions=[
                Condition(field="confidence", operator="greater_than", value=0.8)
            ],
            actions=[Action(type="highlight", parameters={})],
            created_at=datetime.now(),
            active=True
        )
        
        result = await self.executor._evaluate_single_rule(rule, analysis_data)
        
        assert result["executed"] is False
        assert "reason" in result
    
    def test_evaluate_condition_greater_than(self):
        """Test evaluating greater_than condition."""
        condition = Condition(field="confidence", operator="greater_than", value=0.8)
        
        assert self.executor._evaluate_condition(condition, {"confidence": 0.9}) is True
        assert self.executor._evaluate_condition(condition, {"confidence": 0.7}) is False
    
    def test_evaluate_condition_less_than(self):
        """Test evaluating less_than condition."""
        condition = Condition(field="position_x", operator="less_than", value=100)
        
        assert self.executor._evaluate_condition(condition, {"position_x": 50}) is True
        assert self.executor._evaluate_condition(condition, {"position_x": 150}) is False
    
    def test_evaluate_condition_equals(self):
        """Test evaluating equals condition."""
        condition = Condition(field="status", operator="equals", value="detected")
        
        assert self.executor._evaluate_condition(condition, {"status": "detected"}) is True
        assert self.executor._evaluate_condition(condition, {"status": "not_detected"}) is False
    
    def test_evaluate_condition_contains(self):
        """Test evaluating contains condition."""
        condition = Condition(field="tags", operator="contains", value="mug")
        
        assert self.executor._evaluate_condition(condition, {"tags": ["mug", "kitchen"]}) is True
        assert self.executor._evaluate_condition(condition, {"tags": ["cup", "table"]}) is False
    
    def test_evaluate_condition_missing_field(self):
        """Test evaluating condition with missing field."""
        condition = Condition(field="missing_field", operator="equals", value="test")
        
        assert self.executor._evaluate_condition(condition, {"other_field": "value"}) is False
    
    @pytest.mark.asyncio
    async def test_execute_actions(self):
        """Test executing actions."""
        actions = [
            Action(type="highlight", parameters={"color": "red"}),
            Action(type="send_notification", parameters={"message": "Mug detected"})
        ]
        
        with patch.object(self.executor, '_execute_highlight_action') as mock_highlight, \
             patch.object(self.executor, '_execute_notification_action') as mock_notify:
            
            result = await self.executor._execute_actions(actions)
            
            assert "highlight" in result
            assert "send_notification" in result
            mock_highlight.assert_called_once()
            mock_notify.assert_called_once()


@pytest.mark.unit
class TestRuleStorage(AsyncBaseTest):
    """Test cases for RuleStorage."""
    
    async def setup_method_async(self):
        """Setup storage test."""
        await super().setup_method_async()
        
        self.mock_db = AsyncMock()
        self.storage = RuleStorage(database=self.mock_db)
    
    @pytest.mark.asyncio
    async def test_save_rule_success(self):
        """Test saving rule successfully."""
        rule = Rule(
            id="rule_123",
            name="Test Rule",
            rule_type=RuleType.CONDITIONAL,
            conditions=[],
            actions=[],
            created_at=datetime.now(),
            active=True
        )
        
        self.mock_db.rules.insert_one.return_value = AsyncMock(inserted_id="rule_123")
        
        result = await self.storage.save_rule(rule)
        
        assert result == "rule_123"
        self.mock_db.rules.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_rule_success(self):
        """Test getting rule successfully."""
        rule_data = {
            "_id": "rule_123",
            "name": "Test Rule",
            "rule_type": "conditional",
            "conditions": [],
            "actions": [],
            "created_at": datetime.now(),
            "active": True
        }
        
        self.mock_db.rules.find_one.return_value = rule_data
        
        result = await self.storage.get_rule("rule_123")
        
        assert result is not None
        assert result.id == "rule_123"
        assert result.name == "Test Rule"
    
    @pytest.mark.asyncio
    async def test_get_rule_not_found(self):
        """Test getting non-existent rule."""
        self.mock_db.rules.find_one.return_value = None
        
        result = await self.storage.get_rule("nonexistent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_list_rules(self):
        """Test listing rules."""
        rules_data = [
            {
                "_id": "rule_1",
                "name": "Rule 1", 
                "rule_type": "conditional",
                "conditions": [],
                "actions": [],
                "created_at": datetime.now(),
                "active": True
            },
            {
                "_id": "rule_2",
                "name": "Rule 2",
                "rule_type": "conditional", 
                "conditions": [],
                "actions": [],
                "created_at": datetime.now(),
                "active": False
            }
        ]
        
        mock_cursor = AsyncMock()
        mock_cursor.to_list.return_value = rules_data
        self.mock_db.rules.find.return_value = mock_cursor
        
        result = await self.storage.list_rules()
        
        assert len(result) == 2
        assert result[0].id == "rule_1"
        assert result[1].id == "rule_2"
    
    @pytest.mark.asyncio
    async def test_update_rule_success(self):
        """Test updating rule successfully."""
        updates = {"name": "Updated Rule", "active": False}
        
        self.mock_db.rules.update_one.return_value = AsyncMock(modified_count=1)
        
        result = await self.storage.update_rule("rule_123", updates)
        
        assert result is True
        self.mock_db.rules.update_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_rule_success(self):
        """Test deleting rule successfully."""
        self.mock_db.rules.delete_one.return_value = AsyncMock(deleted_count=1)
        
        result = await self.storage.delete_rule("rule_123")
        
        assert result is True
        self.mock_db.rules.delete_one.assert_called_once()


@pytest.mark.unit
class TestRuleModels:
    """Test cases for rule data models."""
    
    def test_create_rule(self):
        """Test creating Rule model."""
        rule = Rule(
            id="rule_123",
            name="Test Rule",
            rule_type=RuleType.CONDITIONAL,
            conditions=[
                Condition(field="confidence", operator="greater_than", value=0.8)
            ],
            actions=[
                Action(type="highlight", parameters={"color": "red"})
            ],
            created_at=datetime.now(),
            active=True
        )
        
        assert rule.id == "rule_123"
        assert rule.name == "Test Rule"
        assert rule.rule_type == RuleType.CONDITIONAL
        assert len(rule.conditions) == 1
        assert len(rule.actions) == 1
        assert rule.active is True
    
    def test_create_condition(self):
        """Test creating Condition model."""
        condition = Condition(
            field="confidence",
            operator="greater_than",
            value=0.8
        )
        
        assert condition.field == "confidence"
        assert condition.operator == "greater_than"
        assert condition.value == 0.8
    
    def test_create_action(self):
        """Test creating Action model."""
        action = Action(
            type="send_alert",
            parameters={"recipient": "admin@example.com", "priority": "high"}
        )
        
        assert action.type == "send_alert"
        assert action.parameters["recipient"] == "admin@example.com"
        assert action.parameters["priority"] == "high"
    
    def test_rule_to_dict(self):
        """Test converting Rule to dictionary."""
        rule = Rule(
            id="rule_123",
            name="Test Rule",
            rule_type=RuleType.CONDITIONAL,
            conditions=[
                Condition(field="confidence", operator="greater_than", value=0.8)
            ],
            actions=[
                Action(type="highlight", parameters={"color": "red"})
            ],
            created_at=datetime.now(),
            active=True
        )
        
        rule_dict = rule.to_dict()
        
        assert rule_dict["id"] == "rule_123"
        assert rule_dict["name"] == "Test Rule"
        assert len(rule_dict["conditions"]) == 1
        assert len(rule_dict["actions"]) == 1