import main_testing
import os

for lookup_table in os.listdir("../lookup_tables"):
    main_testing.main("10", 100, "../lookup_tables/" + lookup_table)
