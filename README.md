# NCS_project - Ventilation effects in Multi-zonal spaces on airborne infection

## Code structure
![Code structure](https://github.com/thigton/NCS_project/blob/master/code_structure.png)

## CX1 workflow
1. Write run script as you want, (test on local machine)
2. "sh send_file_to_ssh.sh <enter_run_name_here> <enter_python_script_to_run_here>"
3. log onto CX1 ssh -XY <username>@login.hpc.ic.ac.uk
4. update job file for requirements
5. "qsub job"
6. Once finished (on CX1): "sh tidy_up_after_cleaning.sh <enter_run_name_here>"
7. On local machine "sh retrieve_results_from_cx1.sh <enter_run_name_here>"
8. Remove folder on CX1 if you need the space
