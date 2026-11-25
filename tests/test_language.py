import unittest
from report_language import comparative_report
class LanguageTests(unittest.TestCase):
    def test_english_boundaries_and_prefix(self):
        self.assertTrue(comparative_report('Effusion redemonstrated.'))
        self.assertFalse(comparative_report('Renewed examination.'))
    def test_spanish_literal_no_accent_folding(self):
        self.assertTrue(comparative_report('Sin cambios.','es'))
        self.assertFalse(comparative_report('Mejoría.','es'))
    def test_missing_is_not_negative(self):
        with self.assertRaises(ValueError):comparative_report('')
