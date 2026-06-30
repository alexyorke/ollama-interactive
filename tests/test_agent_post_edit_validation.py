import unittest

from tests.agent_test_support import AgentTestBase
from tests.test_agent import EXTRACTED_POST_EDIT_VALIDATION_TESTS


class AgentPostEditValidationTests(AgentTestBase):
    pass


for _name, _test in EXTRACTED_POST_EDIT_VALIDATION_TESTS.items():
    setattr(AgentPostEditValidationTests, _name, _test)
