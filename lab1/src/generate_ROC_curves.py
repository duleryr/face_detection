import main_testing
import os

for lookup_table in os.listdir("../lookup_tables"):
    os.system("python3 main_testing.py 10 2 ../lookup_tables/"+str(lookup_table))
