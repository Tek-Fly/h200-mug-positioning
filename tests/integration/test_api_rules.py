"""Integration tests for rules API endpoints."""

import pytest
import json
from datetime import datetime
from unittest.mock import patch, AsyncMock
from fastapi import status

from src.core.rules.models import Rule, RuleType, Condition, Action
from tests.base import APITestBase, IntegrationTestBase


@pytest.mark.integration
class TestRulesAPI(APITestBase, IntegrationTestBase):
    """Test cases for rules API endpoints."""
    
    async def setup_method_async(self):
        """Setup rules API tests."""
        await super().setup_method_async()
        await self.mock_database_operations()
        
        # Sample rule data
        self.sample_rule_data = {
            "name": "High Confidence Alert",
            "rule_type": "conditional",
            "conditions": [
                {
                    "field": "confidence",
                    "operator": "greater_than",
                    "value": 0.8
                }
            ],
            "actions": [
                {
                    "type": "send_alert",
                    "parameters": {"message": "High confidence detection"}
                }
            ],
            "active": True
        }
    
    def test_create_rule_success(self, test_client, auth_headers):
        """Test successful rule creation."""
        with patch('src.core.rules.engine.RulesEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.create_rule.return_value = "rule_123"
            mock_engine_class.return_value = mock_engine
            
            response = test_client.post(
                f"{self.base_url}/rules",
                json=self.sample_rule_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            
            assert data["id"] == "rule_123"
            assert "message" in data
            mock_engine.create_rule.assert_called_once()
    
    def test_create_rule_from_natural_language_success(self, test_client, auth_headers):
        """Test creating rule from natural language."""
        rule_text = "If mug confidence is greater than 0.8, then send alert"
        
        with patch('src.core.rules.engine.RulesEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.create_rule_from_text.return_value = "rule_123"
            mock_engine_class.return_value = mock_engine
            
            response = test_client.post(
                f"{self.base_url}/rules/natural-language",
                json={"rule_text": rule_text},
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            
            assert data["id"] == "rule_123"
            mock_engine.create_rule_from_text.assert_called_once_with(rule_text)
    
    def test_create_rule_invalid_syntax(self, test_client, auth_headers):
        """Test creating rule with invalid syntax."""
        rule_text = "This is not a valid rule syntax"
        
        with patch('src.core.rules.engine.RulesEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.create_rule_from_text.side_effect = ValueError("Invalid rule syntax")
            mock_engine_class.return_value = mock_engine
            
            response = test_client.post(
                f"{self.base_url}/rules/natural-language",
                json={"rule_text": rule_text},
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
            data = response.json()
            self.assert_error_response(data, "INVALID_RULE_SYNTAX")
    
    def test_get_rule_success(self, test_client, auth_headers):
        """Test getting existing rule."""
        rule_id = "rule_123"
        
        sample_rule = Rule(
            id=rule_id,
            name="Test Rule",
            rule_type=RuleType.CONDITIONAL,
            conditions=[
                Condition(field="confidence", operator="greater_than", value=0.8)
            ],
            actions=[
                Action(type="alert", parameters={})
            ],
            created_at=datetime.now(),
            active=True
        )
        
        with patch('src.core.rules.engine.RulesEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.get_rule.return_value = sample_rule
            mock_engine_class.return_value = mock_engine
            
            response = test_client.get(
                f"{self.base_url}/rules/{rule_id}",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["id"] == rule_id
            assert data["name"] == "Test Rule"
            assert data["active"] is True
            assert len(data["conditions"]) == 1
            assert len(data["actions"]) == 1
    
    def test_get_rule_not_found(self, test_client, auth_headers):
        """Test getting non-existent rule."""
        rule_id = "nonexistent_rule"
        
        with patch('src.core.rules.engine.RulesEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.get_rule.return_value = None
            mock_engine_class.return_value = mock_engine
            
            response = test_client.get(
                f"{self.base_url}/rules/{rule_id}",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_list_rules_success(self, test_client, auth_headers):
        """Test listing rules."""
        sample_rules = [
            Rule(
                id="rule_1",
                name="Rule 1",
                rule_type=RuleType.CONDITIONAL,
                conditions=[],
                actions=[],
                created_at=datetime.now(),
                active=True
            ),
            Rule(
                id="rule_2",
                name="Rule 2",
                rule_type=RuleType.CONDITIONAL,
                conditions=[],
                actions=[],
                created_at=datetime.now(),
                active=False
            )
        ]
        
        with patch('src.core.rules.engine.RulesEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.list_rules.return_value = sample_rules
            mock_engine_class.return_value = mock_engine
            
            response = test_client.get(
                f"{self.base_url}/rules",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "items" in data
            assert len(data["items"]) == 2
            assert data["items"][0]["id"] == "rule_1"
            assert data["items"][1]["id"] == "rule_2"
    
    def test_list_rules_filtered(self, test_client, auth_headers):
        """Test listing rules with filters."""
        active_rules = [
            Rule(
                id="rule_1",
                name="Active Rule",
                rule_type=RuleType.CONDITIONAL,
                conditions=[],
                actions=[],
                created_at=datetime.now(),
                active=True
            )
        ]
        
        with patch('src.core.rules.engine.RulesEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.list_rules.return_value = active_rules
            mock_engine_class.return_value = mock_engine
            
            response = test_client.get(
                f"{self.base_url}/rules",
                params={"active_only": True},
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert len(data["items"]) == 1
            assert data["items"][0]["active"] is True
            mock_engine.list_rules.assert_called_once_with(active_only=True)
    
    def test_update_rule_success(self, test_client, auth_headers):
        """Test updating existing rule."""
        rule_id = "rule_123"
        updates = {"name": "Updated Rule Name", "active": False}
        
        with patch('src.core.rules.engine.RulesEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.update_rule.return_value = True
            mock_engine_class.return_value = mock_engine
            
            response = test_client.patch(
                f"{self.base_url}/rules/{rule_id}",
                json=updates,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "message" in data
            mock_engine.update_rule.assert_called_once_with(rule_id, updates)
    
    def test_update_rule_not_found(self, test_client, auth_headers):
        """Test updating non-existent rule."""
        rule_id = "nonexistent_rule"
        updates = {"name": "Updated Name"}
        
        with patch('src.core.rules.engine.RulesEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.update_rule.return_value = False
            mock_engine_class.return_value = mock_engine
            
            response = test_client.patch(
                f"{self.base_url}/rules/{rule_id}",
                json=updates,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_delete_rule_success(self, test_client, auth_headers):
        """Test deleting rule."""
        rule_id = "rule_123"
        
        with patch('src.core.rules.engine.RulesEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.delete_rule.return_value = True
            mock_engine_class.return_value = mock_engine
            
            response = test_client.delete(
                f"{self.base_url}/rules/{rule_id}",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_204_NO_CONTENT
            mock_engine.delete_rule.assert_called_once_with(rule_id)
    
    def test_delete_rule_not_found(self, test_client, auth_headers):
        """Test deleting non-existent rule."""
        rule_id = "nonexistent_rule"
        
        with patch('src.core.rules.engine.RulesEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.delete_rule.return_value = False
            mock_engine_class.return_value = mock_engine
            
            response = test_client.delete(
                f"{self.base_url}/rules/{rule_id}",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_evaluate_rules_success(self, test_client, auth_headers):
        """Test evaluating rules against analysis data."""
        analysis_data = {
            "confidence": 0.9,
            "position_x": 150,
            "detections": [{"class": "cup", "confidence": 0.9}]
        }
        
        evaluation_results = [
            {
                "rule_id": "rule_1",
                "executed": True,
                "actions": ["send_alert"],
                "reason": "Confidence threshold met"
            },
            {
                "rule_id": "rule_2",
                "executed": False,
                "reason": "Condition not met"
            }
        ]
        
        with patch('src.core.rules.engine.RulesEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.evaluate_rules.return_value = evaluation_results
            mock_engine_class.return_value = mock_engine
            
            response = test_client.post(
                f"{self.base_url}/rules/evaluate",
                json=analysis_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "results" in data
            assert len(data["results"]) == 2
            assert data["results"][0]["executed"] is True
            assert data["results"][1]["executed"] is False
            
            mock_engine.evaluate_rules.assert_called_once_with(analysis_data)
    
    def test_validate_rule_success(self, test_client, auth_headers):
        """Test rule validation."""
        with patch('src.core.rules.engine.RulesEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.validate_rule.return_value = True
            mock_engine_class.return_value = mock_engine
            
            response = test_client.post(
                f"{self.base_url}/rules/validate",
                json=self.sample_rule_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["valid"] is True
            assert "message" in data
    
    def test_validate_rule_invalid(self, test_client, auth_headers):
        """Test invalid rule validation."""
        invalid_rule = self.sample_rule_data.copy()
        invalid_rule["conditions"] = []  # Invalid: no conditions
        
        with patch('src.core.rules.engine.RulesEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.validate_rule.side_effect = ValueError("Rule must have conditions")
            mock_engine_class.return_value = mock_engine
            
            response = test_client.post(
                f"{self.base_url}/rules/validate",
                json=invalid_rule,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK  # Validation endpoint returns 200
            data = response.json()
            
            assert data["valid"] is False
            assert "error" in data
    
    def test_get_rule_execution_history(self, test_client, auth_headers):
        """Test getting rule execution history."""
        rule_id = "rule_123"
        
        execution_history = [
            {
                "execution_id": "exec_1",
                "timestamp": "2025-01-01T10:00:00Z",
                "executed": True,
                "analysis_id": "analysis_1",
                "actions_performed": ["send_alert"]
            },
            {
                "execution_id": "exec_2",
                "timestamp": "2025-01-01T11:00:00Z",
                "executed": False,
                "analysis_id": "analysis_2",
                "reason": "Condition not met"
            }
        ]
        
        with patch('src.database.mongodb.get_mongodb_client') as mock_get_db:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_cursor = AsyncMock()
            
            mock_cursor.to_list.return_value = execution_history
            mock_collection.find.return_value = mock_cursor
            mock_db.__getitem__.return_value = mock_collection
            mock_get_db.return_value = mock_db
            
            response = test_client.get(
                f"{self.base_url}/rules/{rule_id}/history",
                params={"limit": 10},
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "items" in data
            assert len(data["items"]) == 2
            assert data["items"][0]["executed"] is True
            assert data["items"][1]["executed"] is False
    
    def test_get_rule_statistics(self, test_client, auth_headers):
        """Test getting rule execution statistics."""
        rule_id = "rule_123"
        
        statistics = {
            "total_evaluations": 100,
            "executions": 75,
            "execution_rate": 0.75,
            "last_executed": "2025-01-01T12:00:00Z",
            "avg_execution_time_ms": 50,
            "most_common_actions": ["send_alert", "highlight"]
        }
        
        # Mock database query for statistics
        with patch('src.database.mongodb.get_mongodb_client') as mock_get_db:
            # This would involve complex aggregation queries in real implementation
            # For testing, we'll mock the computed result
            mock_db = AsyncMock()
            
            # Mock the statistics calculation
            with patch('src.core.rules.engine.RulesEngine') as mock_engine_class:
                mock_engine = AsyncMock()
                mock_engine.get_rule_statistics.return_value = statistics
                mock_engine_class.return_value = mock_engine
                
                response = test_client.get(
                    f"{self.base_url}/rules/{rule_id}/statistics",
                    headers=auth_headers
                )
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                
                assert data["total_evaluations"] == 100
                assert data["execution_rate"] == 0.75
                assert "last_executed" in data
    
    def test_create_rule_missing_required_fields(self, test_client, auth_headers):
        """Test creating rule with missing required fields."""
        incomplete_rule = {
            "name": "Incomplete Rule"
            # Missing rule_type, conditions, actions
        }
        
        response = test_client.post(
            f"{self.base_url}/rules",
            json=incomplete_rule,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_bulk_rule_operations(self, test_client, auth_headers):
        """Test bulk rule operations."""
        bulk_operations = {
            "operations": [
                {"action": "activate", "rule_id": "rule_1"},
                {"action": "deactivate", "rule_id": "rule_2"},
                {"action": "delete", "rule_id": "rule_3"}
            ]
        }
        
        with patch('src.core.rules.engine.RulesEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.bulk_update_rules.return_value = {
                "success": 2,
                "failed": 1,
                "details": [
                    {"rule_id": "rule_1", "status": "success"},
                    {"rule_id": "rule_2", "status": "success"},
                    {"rule_id": "rule_3", "status": "failed", "error": "Rule not found"}
                ]
            }
            mock_engine_class.return_value = mock_engine
            
            response = test_client.post(
                f"{self.base_url}/rules/bulk",
                json=bulk_operations,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["success"] == 2
            assert data["failed"] == 1
            assert len(data["details"]) == 3