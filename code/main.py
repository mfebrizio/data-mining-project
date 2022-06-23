# import package
from runpy import run_path


# Function to call project scripts
def run_project_code():
    run_path("retrieve_FR_data.py")
    run_path("data_preprocessing.py")
    run_path("EDA.py")
    run_path("modeling_1.py")
    run_path("modeling_2.py")
    run_path("modeling_3.py")
    run_path("modeling_4.py")


# Run the script
if __name__ == '__main__':
    run_project_code()
