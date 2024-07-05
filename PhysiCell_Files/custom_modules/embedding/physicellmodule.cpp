////////
// title: PhyiCell/custom_modules/physicellmodule.cpp
// derived from: PhyiCell/main.cpp
//
// language: C/C++
// date: 2015-2024
// license: BSD-3-Clause
// author: Alexandre Bertin, Elmar Bucher, Paul Macklin
// original source code: https://github.com/MathCancer/PhysiCell
// modified source code: https://github.com/elmbeech/physicellembedding
// modified source code: https://github.com/Dante-Berth/PhysiGym
// input: https://docs.python.org/3/extending/extending.html
////////


// pull in the python API
// since Python may define some pre-processor definitions which affect the standard headers on some systems, you must include Python.h before any standard headers are included.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// load standard libraries
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <omp.h>

// loade physicell libraries
#include "core/PhysiCell.h"
#include "modules/PhysiCell_standard_modules.h"
#include "../../custom_modules/custom.h"

// load namespace
using namespace BioFVM;
using namespace PhysiCell;

// global variables
char filename[1024];
std::ofstream report_file;
//std::vector<std::string> (*cell_coloring_function)(Cell*) = my_coloring_function;


// functions

// extended python3 C++ function start
static PyObject* physicell_start(PyObject *self, PyObject *args) {

    // extract args take default if no args
    char *argument = "config/PhysiCell_settings.xml";
    if (!PyArg_ParseTuple(args, "s", &argument)) { PyErr_Clear(); }

    // load and parse settings file (modules/PhysiCell_settings.cpp).
    bool XML_status = false;
    XML_status = load_PhysiCell_config_file(argument, true);  // (over)load parameter and density definitions
    if (!XML_status) { exit(-1); }

    // copy config file to output directory
    char copy_command [1024];
    sprintf(copy_command, "cp %s %s", argument, PhysiCell_settings.folder.c_str());
    system(copy_command);

    // copy seeding file
    // nop

    // reset global variables
    PhysiCell_globals = PhysiCell_Globals();

    // OpenMP setup
    omp_set_num_threads(PhysiCell_settings.omp_num_threads);

    // time setup
    std::string time_units = "min";

    // Microenvironment setup //
    setup_microenvironment(); // modify this in the custom code

    // PhysiCell setup ///

    // set mechanics voxel size, and match the data structure to BioFVM
    double mechanics_voxel_size = 30;
    Cell_Container* cell_container = create_cell_container_for_microenvironment(microenvironment, mechanics_voxel_size);

    // Users typically start modifying here.
    random_seed();
    generate_cell_types();  // delete cells; reload cell definitions; display_cell_definitions; generates rules.csv v1 file
    setup_tissue();
    // Users typically stop modifying here.

    // set MultiCellDS save options
    set_save_biofvm_mesh_as_matlab(true);
    set_save_biofvm_data_as_matlab(true);
    set_save_biofvm_cell_data(true);
    set_save_biofvm_cell_data_as_custom_matlab(true);

    // save data initial simulation snapshot
    //char filename[1024];  // bue 20240130: going global
    sprintf(filename, "%s/initial", PhysiCell_settings.folder.c_str());
    save_PhysiCell_to_MultiCellDS_v2(filename, microenvironment, PhysiCell_globals.current_time);

    // save data output00000000 simulation snapshot
    if (PhysiCell_settings.enable_full_saves == true ) {
        sprintf(filename, "%s/output%08u", PhysiCell_settings.folder.c_str(),  PhysiCell_globals.full_output_index);
        save_PhysiCell_to_MultiCellDS_v2(filename, microenvironment, PhysiCell_globals.current_time);
    }

    // save legend.svg and initial.svg cross section through z = 0
    PhysiCell_SVG_options.length_bar = 200;  // after setting cross section length bar to 200 microns
    std::vector<std::string> (*cell_coloring_function)(Cell*) = my_coloring_function;  // set pathology coloring function // bue 20240130: not going global

    sprintf(filename, "%s/legend.svg", PhysiCell_settings.folder.c_str());
    create_plot_legend(filename, cell_coloring_function);

    sprintf(filename, "%s/initial.svg", PhysiCell_settings.folder.c_str());
    SVG_plot(filename, microenvironment, 0.0, PhysiCell_globals.current_time, cell_coloring_function);

    // save snapshot00000000.svg cross section through z = 0
    if (PhysiCell_settings.enable_SVG_saves == true) {
        sprintf(filename, "%s/snapshot%08u.svg", PhysiCell_settings.folder.c_str(), PhysiCell_globals.SVG_output_index);
        SVG_plot(filename, microenvironment, 0.0, PhysiCell_globals.current_time, cell_coloring_function);
    }

    // save legacy simulation_report
    //std::ofstream report_file;  // bue 20240130: going global
    if (PhysiCell_settings.enable_legacy_saves == true) {
        sprintf(filename, "%s/simulation_report.txt", PhysiCell_settings.folder.c_str());
        report_file.open(filename);  // create the data log file
        report_file << "simulated time\tnum cells\tnum division\tnum death\twall time" << std::endl;
        log_output(PhysiCell_globals.current_time, PhysiCell_globals.full_output_index, microenvironment, report_file);
    }

    // standard outout
    display_citations();
    display_simulation_status(std::cout);  // output00000000

    // set the performance timers
    BioFVM::RUNTIME_TIC();
    BioFVM::TIC();

    // going home
    return PyLong_FromLong(0);
}


// extended python3 C++ function step
static PyObject* physicell_step(PyObject *self, PyObject *args) {
    // main loop

    // bue 2024-02-01: simplify (no try catch)
    //try {

        // set time variables
        // achtung : begin physigym specific implementation!
        double custom_countdown = parameters.doubles("dt_gym");  // [min]
        // achtung : end physigym specific implementation!
        double phenotype_countdown = phenotype_dt;
        double mechanics_countdown = mechanics_dt;
        double mcds_countdown = PhysiCell_settings.full_save_interval;
        double svg_countdown = PhysiCell_settings.SVG_save_interval;

        // run diffusion time step paced main loop
        bool action = true;
        bool step = true;
        while (step) {

            // max time reached?
            if (PhysiCell_globals.current_time > (PhysiCell_settings.max_time + parameters.doubles("dt_gym"))) {
                step = false;
            }

            // do action
            if (action) {

                // achtung : begin physigym specific implementation!
                std::cout << "processing action block ... " << std::endl;
                action = false;
                // achtung : end physigym specific implementation!

                // Put physigym related parameter, variable, and vector action mapping here!

                // parameter
                //my_function( parameters.bools("my_bool")) );
                //my_function( parameters.ints("my_int")) );
                //my_function( parameters.doubles("my_float") );
                //my_function( parameters.strings("my_str") );

                // custom variable
                //std::string my_variable = "my_variable";
                //for (Cell* pCell : (*all_cells)) {
                //    my_function( pCell->custom_data[my_variable] );
                //}

                // custom vector
                //std::string my_vector = "my_vector";
                //for (Cell* pCell : (*all_cells)) {
                //    int vectindex = pCell->custom_data.find_vector_variable_index(my_vector);
                //    if (vectindex > -1) {
                //        my_function( pCell->custom_data.vector_variables[vectindex].value );
                //    } else {
                //        char error[64];
                //        sprintf(error, "Error: unknown custom_data vector! %s", my_vector);
                //        PyErr_SetString(PyExc_ValueError, error);
                //        return NULL;
                //    }
                //}

                // update substrate concentration
                set_microenv("drug", parameters.doubles("drug_dose"));
            }

            // do observation
            // on dt_gym time step
            if (custom_countdown < diffusion_dt / 3) {

                // achtung : begin physigym specific implementation!
                custom_countdown += parameters.doubles("dt_gym");  // [min]
                std::cout << "processing gym time step observation block ... " << std::endl;
                parameters.doubles("time") = PhysiCell_globals.current_time;
                action = true;
                step = false;
                // achtung : end physigym specific implementation!

                // Put physigym related parameter, variable, and vector observation mapping here!

                // parameter
                //parameters.bools("my_bool") = value;
                //parameters.ints("my_int") = value;
                //parameters.doubles("my_float") = value;
                //parameters.strings("my_str") = value;

                // custom variable
                //std::string my_variable = "my_variable";
                //for (Cell* pCell : (*all_cells)) {
                //    pCell->custom_data[my_variable] = value;
                //}

                // custom vector
                //std::string my_vector = "my_vector";
                //for (Cell* pCell : (*all_cells)) {
                //    int vectindex = pCell->custom_data.find_vector_variable_index(my_vector);
                //    if (vectindex > -1) {
                //        pCell->custom_data.vector_variables[vectindex].value = value;
                //    } else {
                //        char error[64];
                //        sprintf(error, "Error: unknown custom_data vector! %s", my_vector);
                //        PyErr_SetString(PyExc_ValueError, error);
                //        return NULL;
                //    }
                //}

                // receive cell count
                parameters.ints("cell_count") = (*all_cells).size();

                // receive apoptosis rate
                for (Cell* pCell : (*all_cells)) {
                    pCell->custom_data["apoptosis_rate"] = get_single_behavior(pCell, "apoptosis");
                }
            }

            // on phenotype time step
            if (phenotype_countdown < diffusion_dt / 3) {
                phenotype_countdown += phenotype_dt;

                // Put phenotype time scale code here!
                //std::cout << "processing phenotype time step observation block ... " << std::endl;
                //step = false;
            }

            // on mechanics time step
            if (mechanics_countdown < diffusion_dt / 3) {
                mechanics_countdown += mechanics_dt;

                // Put mechanics time scale code here!
                //std::cout << "processing mechanic time step observation block ... " << std::endl;
                //step = false;
            }

            // on diffusion time step

            // Put diffusion time scale code here!
            //std::cout << "processing diffusion time step observation block ... " << std::endl << std::endl;
            //step = false;

            // run microenvironment
            microenvironment.simulate_diffusion_decay(diffusion_dt);

            // run PhysiCell
            ((Cell_Container *)microenvironment.agent_container)->update_all_cells(PhysiCell_globals.current_time);

            // update time
            custom_countdown -= diffusion_dt;
            phenotype_countdown -= diffusion_dt;
            mechanics_countdown -= diffusion_dt;
            mcds_countdown -= diffusion_dt;
            svg_countdown -= diffusion_dt;
            PhysiCell_globals.current_time += diffusion_dt;

            // save data if it's time.
            if (mcds_countdown < diffusion_dt / 3) {
                mcds_countdown += PhysiCell_settings.full_save_interval;
                PhysiCell_globals.full_output_index++;

                display_simulation_status(std::cout);

                if (PhysiCell_settings.enable_legacy_saves == true) {
                    log_output(PhysiCell_globals.current_time, PhysiCell_globals.full_output_index, microenvironment, report_file);
                }

                if (PhysiCell_settings.enable_full_saves == true ) {
                    sprintf(filename, "%s/output%08u", PhysiCell_settings.folder.c_str(),  PhysiCell_globals.full_output_index);
                    save_PhysiCell_to_MultiCellDS_v2(filename, microenvironment, PhysiCell_globals.current_time);
                }
            }

            // save svg plot if it's time
            if ((PhysiCell_settings.enable_SVG_saves == true) and (svg_countdown < diffusion_dt / 3)) {
                svg_countdown += PhysiCell_settings.SVG_save_interval;
                PhysiCell_globals.SVG_output_index++;

                std::vector<std::string> (*cell_coloring_function)(Cell*) = my_coloring_function;  // bue 20240130: not going global
                sprintf(filename, "%s/snapshot%08u.svg", PhysiCell_settings.folder.c_str(), PhysiCell_globals.SVG_output_index);
                SVG_plot(filename, microenvironment, 0.0, PhysiCell_globals.current_time, cell_coloring_function);
            }
        }

        // save legacy simulation report
        if (PhysiCell_settings.enable_legacy_saves == true) {
            log_output(PhysiCell_globals.current_time, PhysiCell_globals.full_output_index, microenvironment, report_file);
            report_file.close();
        }

    //} catch (const std::exception& e) {
        // reference to the base of a polymorphic object
    //    std::cout << e.what(); // information from length_error printed
    //}

    // go home
    return PyLong_FromLong(0);
}


// extended python3 C++ function stop
static PyObject* physicell_stop(PyObject *self, PyObject *args) {

    // save final simulation snapshot
    sprintf(filename, "%s/final", PhysiCell_settings.folder.c_str());
    save_PhysiCell_to_MultiCellDS_v2(filename, microenvironment, PhysiCell_globals.current_time);

    std::vector<std::string> (*cell_coloring_function)(Cell*) = my_coloring_function;  // bue 20240130: not going global
    sprintf(filename, "%s/final.svg", PhysiCell_settings.folder.c_str());
    SVG_plot(filename, microenvironment, 0.0, PhysiCell_globals.current_time, cell_coloring_function);

    // timer
    std::cout << "Total simulation runtime: " << std::endl;
    BioFVM::display_stopwatch_value(std::cout, BioFVM::runtime_stopwatch_value());
    std::cout << std::endl;

    // go home
    return PyLong_FromLong(0);
}


// extend python3 C++ function set_parameter
static PyObject* physicell_set_parameter(PyObject *self, PyObject *args) {
    // extract input
    const char *label;
    PyObject *value;
    if (! PyArg_ParseTuple(args, "sO", &label, &value)) {
        return NULL;
    }

    // recall from C++ into python3 variable
    int parindex;
    char error[64];

    // boole
    parindex = parameters.bools.find_index(label);
    if (parindex > -1) {
        Parameter<bool> *bool_parameter = &parameters.bools[parindex];
        bool_parameter->value = PyLong_AsLong(value);

    } else {
        // int
        parindex = parameters.ints.find_index(label);
        if (parindex > -1) {
            Parameter<int> *int_parameter = &parameters.ints[parindex];
            int_parameter->value = PyLong_AsLong(value);

        } else {
            // float
            parindex = parameters.doubles.find_index(label);
            if (parindex > -1) {
                Parameter<double> *float_parameter = &parameters.doubles[parindex];
                float_parameter->value = PyFloat_AsDouble(value);

            } else {
                // str
                parindex = parameters.strings.find_index(label);
                if (parindex > -1) {
                    if (! PyUnicode_Check(value)) {
                        sprintf(error, "Error: %s value cannot be interpreted as a string!", label);
                        PyErr_SetString(PyExc_ValueError, error);
                        return NULL;
                    }
                    Parameter<std::string> *str_parameter = &parameters.strings[parindex];
                    str_parameter->value = PyUnicode_AsUTF8(value);

                } else {
                    //error
                    sprintf(error, "Error: unknown parameter! %s", label);
                    PyErr_SetString(PyExc_KeyError, error);
                    return NULL;
                }
            }
        }
    }
    // go home
    return PyLong_FromLong(0);
}


// extend python3 C++ function get_parameter
static PyObject* physicell_get_parameter(PyObject *self, PyObject *args) {
    // extract variable label
    const char *label;
    if (! PyArg_ParseTuple(args, "s", &label)) {
        return NULL;
    }

    // recall from C++ into python3 variable
    int parindex;

    // bool
    parindex = parameters.bools.find_index(label);
    if (parindex > -1) {
        Parameter<bool> bool_parameter = parameters.bools[parindex];
        return PyBool_FromLong(bool_parameter.value);

    } else {
        // int
        parindex = parameters.ints.find_index(label);
        if (parindex > -1) {
            Parameter<int> int_parameter = parameters.ints[parindex];
            return PyLong_FromLong(int_parameter.value);

        } else {
            // float
            parindex = parameters.doubles.find_index(label);
            if (parindex > -1) {
                Parameter<double> float_parameter = parameters.doubles[parindex];
                return PyFloat_FromDouble(float_parameter.value);

            } else {
                // str
                parindex = parameters.strings.find_index(label);
                if (parindex > -1) {
                    Parameter<std::string> str_parameter = parameters.strings[parindex];
                    return PyUnicode_FromString(str_parameter.value.c_str());

                } else {
                    //error
                    char error[64];
                    sprintf(error, "Error: unknown parameter! %s", label);
                    PyErr_SetString(PyExc_KeyError, error);
                    return NULL;
                }
            }
        }
    }
}


// extended python3 C++ function set_variable
static PyObject* physicell_set_variable(PyObject *self, PyObject *args) {
    // extract variable label and value
    const char *label;
    double value;
    if (! PyArg_ParseTuple(args, "sd", &label, &value)) {
        return NULL;
    }

    // store
    for (Cell* pCell : (*all_cells)) {
        //set_single_behavior(pCell, "custom:<label>" , value); // bue20240206: trouble child.
        int varindex = pCell->custom_data.find_variable_index(label);
        if (varindex > -1) {
            pCell->custom_data[label] = value;
        } else {
            char error[64];
            sprintf(error, "Error: unknown custom_data variable! %s", label);
            PyErr_SetString(PyExc_KeyError, error);
            return NULL;
        }
    }

    // going home
    return PyLong_FromLong(0);
}


// extended python3 C++ function get_variable
static PyObject* physicell_get_variable(PyObject *self, PyObject *args) {
    // extract variable label
    const char *label;
    if (! PyArg_ParseTuple(args, "s", &label)) {
        return NULL;
    }

    // recall from C++ intp python3 list
    int cell_count = all_cells->size();
    PyObject *pList = PyList_New(cell_count);
    for (int i=0; i < cell_count; i++) {
        Cell* pCell = (*all_cells)[i];
        int varindex = pCell->custom_data.find_variable_index(label);
        if (varindex > -1) {
            double value = pCell->custom_data[label];
            PyList_SetItem(pList, i, PyFloat_FromDouble(value));
        } else {
            Py_XDECREF(pList);
            char error[64];
            sprintf(error, "Error: unknown custom_data variable! %s", label);
            PyErr_SetString(PyExc_KeyError, error);
            return NULL;
        }
    }

    // going home
    return pList;
}


// extended python3 C++ function set_vector
static PyObject* physicell_set_vector(PyObject *self, PyObject *args) {
    // https://stackoverflow.com/questions/22458298/extending-python-with-c-pass-a-list-to-pyarg-parsetuple
    // https://docs.python.org/3/c-api/arg.html

    // extract variable label and vector
    const char *label;
    PyObject *pList;
    std::vector<double> value;
    if (! PyArg_ParseTuple(args, "sO!", &label, &PyList_Type, &pList)) {
        return NULL;
    }

    // transfrom python list of numbers to C++ vector of doubles
    PyObject *pItem;
    for (int i=0; i < PyList_Size(pList); i++) {
        pItem = PyList_GetItem(pList, i);
        if (!(PyNumber_Check(pItem))) {
            PyErr_SetString(PyExc_TypeError, "Error: all list items must be integer or float!");
            return NULL;
        } else {
            value.push_back(PyFloat_AsDouble(pItem));
        }
    }

    // store
    for (Cell* pCell : (*all_cells)) {
        int vectindex = pCell->custom_data.find_vector_variable_index(label);
        if (vectindex > -1) {
            pCell->custom_data.vector_variables[vectindex].value = value;
        } else {
            char error[64];
            sprintf(error, "Error: unknown custom_data vector! %s", label);
            PyErr_SetString(PyExc_KeyError, error);
            return NULL;
        }
    }

    // going home
    return PyLong_FromLong(0);
}


// extended python3 C++ function get_vector
static PyObject* physicell_get_vector(PyObject *self, PyObject *args) {
    // extract variable label
    const char *label;
    if (! PyArg_ParseTuple(args, "s", &label)) {
        return NULL;
    }

    // recall from C++ into python3 list of list
    bool first = true;
    int cell_count = all_cells->size();
    PyObject *pLlist = PyList_New(cell_count);
    for (int i=0; i < cell_count; i++) {
        Cell* pCell = (*all_cells)[i];
        int vectindex = pCell->custom_data.find_vector_variable_index(label);
        if (first) {
            first = false;
            if (vectindex < 0) {
                Py_XDECREF(pLlist);
                char error[64];
                sprintf(error, "Error: unknown custom_data vector! %s", label);
                PyErr_SetString(PyExc_KeyError, error);
                return NULL;
            }
        }
        std::vector<double> vector = pCell->custom_data.vector_variables[vectindex].value;
        int element_count = vector.size();
        PyObject *pList = PyList_New(element_count);
        for (int j=0; j < element_count; j++) {
            double value = vector[j];
            PyList_SetItem(pList, j, PyFloat_FromDouble(value));
        }
        PyList_SetItem(pLlist, i, pList);
    }

    // going home
    return pLlist;
}


// extended python3 C++ function get_cell
static PyObject* physicell_get_cell(PyObject *self, PyObject *args) {

    // recall from C++ into python3 list of list
    int cell_count = all_cells->size();
    PyObject *pLlist = PyList_New(cell_count);

    for (int i=0; i < cell_count; i++) {
        Cell* pCell = (*all_cells)[i];

        PyObject *pList = PyList_New(4);  // id, x, y, z
        PyList_SetItem(pList, 0, PyLong_FromLong(pCell->ID)); // id
        PyList_SetItem(pList, 1, PyFloat_FromDouble(pCell->position[0])); // x
        PyList_SetItem(pList, 2, PyFloat_FromDouble(pCell->position[1])); // y
        PyList_SetItem(pList, 3, PyFloat_FromDouble(pCell->position[2])); // z
        PyList_SetItem(pLlist, i, pList);
    }

    // going home
    return pLlist;
}


// extended python3 C++ function get_microenv
static PyObject* physicell_get_microenv(PyObject *self, PyObject *args) {
    // extract variable label
    const char *substrate;
    if (! PyArg_ParseTuple(args, "s", &substrate)) {
        return NULL;
    }

    // extract substrate index
    int subsindex = microenvironment.find_density_index(substrate);
    if (subsindex < 0) {
        char error[64];
        sprintf(error, "Error: unknown substrate! %s", substrate);
        PyErr_SetString(PyExc_KeyError, error);
        return NULL;
    }

    // recall from C++ into python3 list of list
    int voxel_count = microenvironment.number_of_voxels();
    PyObject *pLlist = PyList_New(voxel_count);

    for (int n=0; n < voxel_count; n++) {
        PyObject *pList = PyList_New(4);  // x, y, z, conc
        PyList_SetItem(pList, 0, PyFloat_FromDouble(microenvironment.mesh.voxels[n].center[0]));  // x
        PyList_SetItem(pList, 1, PyFloat_FromDouble(microenvironment.mesh.voxels[n].center[1]));  // y
        PyList_SetItem(pList, 2, PyFloat_FromDouble(microenvironment.mesh.voxels[n].center[2]));  // z
        PyList_SetItem(pList, 3, PyFloat_FromDouble(microenvironment(n)[subsindex]));  // conc
        PyList_SetItem(pLlist, n, pList);
    }

    // going home
    return pLlist;
}


// extended python3 C++ function system
static PyObject* physicell_system(PyObject *self, PyObject *args) {
    // variables
    const char *command;
    int sts;

    // extract and run commandline command
    if (! PyArg_ParseTuple(args, "s",  &command)) {
        return NULL;
    }
    sts = system(command);

    // going home
    return PyLong_FromLong(sts);
}


// method table lists method name and address
static struct PyMethodDef ExtendpyMethods[] = {
    {"start", physicell_start, METH_VARARGS,
     "input:\n    args 'path/to/setting.xml' file (string); default is 'config/PhysiCell_settings.xml'.\n\noutput:\n    physicell processing. 0 for success.\n\nrun:\n    from embedding import physicell\n    physicell.start('path/to/setting.xml')\n\ndescription:\n    function (re)initializes physicell as specified in the settings.xml, cells.csv, and cell_rules.csv files and generates the step zero observation output. if run for re-initialization, it is assumed that start will not be called before stop has been called."
    },
    {"step", physicell_step, METH_VARARGS,
     "input:\n    none.\n\noutput:\n    physicell processing. 0 for success.\n\nrun:\n    from embedding import physicell\n    physicell.step()\n\ndescription:\n    function runs one time step."
    },
    {"stop", physicell_stop, METH_VARARGS,
     "input:\n    none.\n\noutput:\n    physicell processing. 0 for success.\n\nrun:\n    from embedding import physicell\n    physicell.stop()\n\ndescription:\n    function finalizes a physicell episode, deletes all the cells, and resets the global variables."
    },
    {"set_parameter", physicell_set_parameter, METH_VARARGS,
     "input:\n    parameter name (string), vector value (bool or int or float or str).\n\noutput:\n    0 for success and -1 for failure.\n\nrun:\n    from embedding import physicell\n    physicell.set_parameter('my_parameter', value)\n\ndescription:\n    function to store a user parameter."
    },
    {"get_parameter", physicell_get_parameter, METH_VARARGS,
     "input:\n    parameter name (string)\n\noutput:\n    values (bool or int or float or str).\n\nrun:\n    from embedding import physicell\n    physicell.get_parameter('my_parameter')\n\ndescription:\n    function to recall a user parameter."
    },
    {"set_variable", physicell_set_variable, METH_VARARGS,
     "input:\n    variable name (string), variable value (float or integer).\n\noutput:\n    0 for success and -1 for failure.\n\nrun:\n    from embedding import physicell\n    physicell.set_variable('my_variable', value)\n\ndescription:\n    function to store a custom variable value."
    },
    {"get_variable", physicell_get_variable, METH_VARARGS,
     "input:\n    variable name (string).\n\noutput:\n    values (list of floats).\n\nrun:\n    from embedding import physicell\n    physicell.get_variable('my_variable')\n\ndescription:\n    function to recall a custom variable."
    },
    {"set_vector", physicell_set_vector, METH_VARARGS,
     "input:\n    vector name (string), vector values (list of floats or integers).\n\noutput:\n    0 for success and -1 for failure.\n\nrun:\n    from embedding import physicell\n    physicell.set_vector('my_vector', value)\n\ndescription:\n    function to store a custom vector."
    },
    {"get_vector", physicell_get_vector, METH_VARARGS,
     "input:\n    vector name (string)\n\noutput:\n    values (list of list of floats).\n\nrun:\n    from embedding import physicell\n    physicell.get_vector('my_vector')\n\ndescription:\n    function to recall a custom vector."
    },
    {"get_cell", physicell_get_cell, METH_VARARGS,
     "input:\n    none\n\noutput:\n    values (list of list of floats).\n\nrun:\n    from embedding import physicell\n    physicell.get_cell()\n\ndescription:\n    function to recall cell position coordinate and id."
    },
    {"get_microenv", physicell_get_microenv, METH_VARARGS,
     "input:\n    substrate name (string)\n\noutput:\n    values (list of list of floats).\n\nrun:\n    from embedding import physicell\n    physicell.get_microenv('my_substrate')\n\ndescription:\n    function to recall a voxel center coordinates and substrate concentration."
    },
    {"system", physicell_system, METH_VARARGS, "execute a shell command."},
    /*{NULL, NULL, 0, NULL}  // Sentinel */
};


// module definition structure
static struct PyModuleDef physicellmodule = {
    PyModuleDef_HEAD_INIT,
    "physicell",  // name of the module
    NULL,  //physicell_doc,  // module documentation, may be NULL
    -1,  // size of per-interpreter state of the module, or -1 if he module keeps state in global variables.
    ExtendpyMethods
};


// pass module structure to the python interpreter initializatin function
PyMODINIT_FUNC PyInit_physicell(void) {
    return PyModule_Create(&physicellmodule);
}


// off we go
int main(int argc, char *argv[]) {
    // get program name
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0].\n");
        exit(1);
    }

    // add the builtin module to the initailzation table
    if (PyImport_AppendInittab("physicell", PyInit_physicell) == -1) {
        fprintf(stderr, "Error: could not extend in-built modules table.\n");
        exit(1);
    }

    // pass argv[0] to the python interpreter
    Py_SetProgramName(program);

    // initialize the python interpreter. required. if this step fails, it will be a fatal error
    Py_Initialize();

    // optional import the module.
    // alternatively, import can be deferred until the embedded script imports it.
    PyObject *pmodule = PyImport_ImportModule("physicell");
    if (!pmodule) {
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'physicell'.\n");
    }

    // free up memory
    PyMem_RawFree(program);
    return 0;
}
