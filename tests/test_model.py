import unittest
from UserDefinedItem import UserDefinedItem

class TestUserDefinedItem(unittest.TestCase): # unittest.TestCase를 상속받음

    def test_generate_vgalue(self):
        item = UserDefinedItem(name='age', item_type='number', options=[10, 30])
        value = item.generate_value()
        self.assertTrue(10 <= value <= 30, "Generated valuue should be within the range.")

    def test_update_parameters(self):
        item = UserDefinedItem(name='interest', item_type='string', options=['tech', 'finance'])
        item.update_parameters(options=['tech', 'finance', 'art'])
        self.assertIn('art', item.options, "Options should be updated with new values.")

    def test_context_based_options(self):
        def context_based_options(context):
            if context['age'] < 18:
                return {"options": [1000, 2000]}
            else:
                return {"options": [2000, 4000]}
            
        item = UserDefinedItem(name='salary', item_type='number', options=[1000, 2000],
                               contextBasedOptions=context_based_options)
        
        young_context = {'age': 15}
        adult_context = {'age': 30}

        young_salary_options = item.evaluate_context_based_options(young_context)
        adult_salary_options = item.evaluate_context_based_options(adult_context)

        self.assertEqual(young_salary_options, {"options": [1000, 2000]}, "Should return options for young context.")
        self.assertEqual(adult_salary_options, {"options": [2000, 4000]}, "Should return options for adult context.")

# Run the tests
if __name__=='__main__':
    unittest.main()