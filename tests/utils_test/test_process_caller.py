import paderbox.utils.process_caller as pc
import unittest

class ProcessCallerTest(unittest.TestCase):

    def test_call_std(self):
        cmds = 5*['echo "Test"']
        stdout, stderr, ret_codes = pc.run_processes(cmds)
        for out in stdout:
            self.assertEqual(out, 'Test\n')
        for err in stderr:
            self.assertEqual(err, '')
        for code in ret_codes:
            self.assertEqual(code, 0)

    def test_call_err(self):
        cmds = 5*['echo "Test" 1>&2']
        stdout, stderr, ret_codes = pc.run_processes(cmds)
        for out in stdout:
            self.assertEqual(out, '')
        for err in stderr:
            self.assertEqual(err, 'Test\n')
        for code in ret_codes:
            self.assertEqual(code, 0)

    def test_debug_mode(self):
        pc.DEBUG_MODE = True
        self.test_call_std()