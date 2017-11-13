import main_testing
import os

# Tests
for lookup_table in os.listdir("../lookup_tables"):
    main_testing.main("10", 2, "../lookup_tables/" + lookup_table)

# C'est parti
for lookup_table in os.listdir("../lookup_tables"):
    main_testing.main("10", 250, "../lookup_tables/" + lookup_table)
