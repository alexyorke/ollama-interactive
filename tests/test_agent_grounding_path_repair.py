import unittest

from tests.agent_test_support import AgentTestBase
from tests.test_agent import EXTRACTED_GROUNDING_PATH_REPAIR_TESTS


class AgentGroundingPathRepairTests(AgentTestBase):
    pass


for _name, _test in EXTRACTED_GROUNDING_PATH_REPAIR_TESTS.items():
    setattr(AgentGroundingPathRepairTests, _name, _test)
