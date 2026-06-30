import unittest

from tests.agent_test_support import AgentTestBase
from tests.test_agent import EXTRACTED_SHELL_COMMAND_PREFLIGHT_TESTS


class AgentShellCommandPreflightTests(AgentTestBase):
    pass


for _name, _test in EXTRACTED_SHELL_COMMAND_PREFLIGHT_TESTS.items():
    setattr(AgentShellCommandPreflightTests, _name, _test)
