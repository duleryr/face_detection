import main_testing
import os

# Tests
for lookup_table in os.listdir("../lookup_tables"):
    os.system("python3 main_testing.py 10 2 ../lookup_tables/"+str(lookup_table))

## C'est parti
#for lookup_table in os.listdir("../lookup_tables"):
#    main_testing.main("10", 250, "../lookup_tables/" + lookup_table)
#>>>>>>> ff345e81e2c485dd9a514117a58c45719c15759d
