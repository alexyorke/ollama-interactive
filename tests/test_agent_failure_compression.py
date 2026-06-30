import unittest

from tests.agent_test_support import AgentTestBase
from tests.test_agent import EXTRACTED_FAILURE_COMPRESSION_TESTS


class AgentFailureCompressionTests(AgentTestBase):
    pass


for _name, _test in EXTRACTED_FAILURE_COMPRESSION_TESTS.items():
    setattr(AgentFailureCompressionTests, _name, _test)
