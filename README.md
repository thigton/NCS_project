# NCS_project - Ventilation effects in Multi-zonal spaces on airborne infection
* The majority of python files in the base directory are run scripts (used to run the model [stored in the 'classes' directory] and store the results in the 'results' directory) or plot scripts.
* There are also a some utility scripts (python and shell) in the base directory too.
* Contam files can be found in 'contam_files'.

## Code structure
The structure about how the different parts of the model are working with each other (all stored in classes) is shown below.
![Code structure](https://github.com/thigton/NCS_project/blob/master/code_structure.png)

 Their purpose, briefly explained
* ContamModel - Handles the interaction with the CONTAM software incl, making changes to the simulation
* ContamPrjSnippets (+subclasses) - Handles the data conversion from the .prj format required for CONTAM to a pandas dataframe.
* ContamVentMatrixStorage - Simple dataclass to store the all the different ventilation matrix. Created so that the simulation parameters used to produce the matrix were tied to the matrices.

* StochasticModel - Collects the results from each simulation / plot overall results.
* Simulation - Handles the running of each CTMC simulation / plot results from individual simulations.
* DTMC_simulation (subclass of Simulation) - Includes changes modifications for the DTMC approach.
* Room - Handles the data associated with a room in the model (will appear as a list in the simulation class).
* Student - Handles the data associated with a group of students in the model (will appear as a list in the simulation class).
* Weather - Handles the weather information (a bit redundant, set up early in case things got more complex i.e wanting to simulation changing weather conditions).

Comment: The structure of the Room/Student class is potentially over complicated for what it is. I set it up this way as I thought it would give the best chance to handle ever increasingly complex ideas about what we could look at with the model.


## CX1 workflow
1. Write run script as you want, (test on local machine)
2. "sh send_file_to_ssh.sh <enter_run_name_here> <enter_python_script_to_run_here>"
3. log onto CX1 ssh -XY <username>@login.hpc.ic.ac.uk
4. update job file for requirements
5. "qsub job"
6. Once finished (on CX1): "sh tidy_up_after_cleaning.sh <enter_run_name_here>"
7. On local machine "sh retrieve_results_from_cx1.sh <enter_run_name_here>"
8. Remove folder on CX1 if you need the space
9. If you are running out of memory on your local machine, move a set of results to an external hard drive by running "move_job_run_to_ext_drive.py" and specify the run name in the terminal command.

## Other useful scripts
  * savepdf_tex.py --- Will take your matplotlib figure and turn it into a .pdf and .pdf_tex file for use in latex. (software requirements: Inkscape)  
