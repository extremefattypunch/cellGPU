#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#include "Simulation.h"
#include "vertexQuadraticEnergyWithNematic.h"
#include "selfPropelledCellVertexDynamics.h"
#include "simpleVertexDatabase.h"
#include <random>
#include <cmath>
#include <vector>
#include <utility>
#include <algorithm>

int main(int argc, char* argv[])
{
    // Default parameters aligned with paper's supplementary Fig. 5
    int numpts = 100; // number of cells
    int USE_GPU = 0; // 0 or greater uses a gpu, any negative number runs on the cpu
    int c;
    int tSteps = 600000; // number of time steps to run after initialization
    int initSteps = 10; // number of initialization steps 

    double dt = 0.01; // the time step size
    double p0 = 3.95;  // the preferred perimeter (fluid phase)
    double a0 = 1.0;  // the preferred area
    double kp = 0.02;  // perimeter elasticity modulus
    double zeta_c = -0.4;  // contractile nematic activity
    double zeta_e = 0.1;  // extensile nematic activity
    double alpha_c = 0.00;  // polar motility for contractile cells
    double alpha_e = 0.00;  // polar motility for contractile cells
    double Dr = 0.0;  // rotational diffusion (deterministic)
    int cluster_size = numpts/2; // central cluster size
    int S = 3;   // 3 for analysis (HDF5 output only)
    int fIdx = 0;

    // The defaults can be overridden from the command line
    while ((c = getopt(argc, argv, "n:g:t:i:p:a:k:c:l:d:s:f:e:")) != -1)
        switch (c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'p': p0 = atof(optarg); break;
            case 'a': a0 = atof(optarg); break;
            case 'k': kp = atof(optarg); break;
            case 'z': zeta_c = atof(optarg); break;
            case 'Z': zeta_e = atof(optarg); break;
            case 'd': Dr = atof(optarg); break;
            case 's': S = atoi(optarg); break;
            case 'f': fIdx = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
            case '?':
                if (optopt == 'c')
                    std::cerr << "Option -" << optopt << " requires an argument.\n";
                return 1;
            default:
                abort();
        };

    clock_t t1, t2;
    bool reproducible = false;
    bool initializeGPU = true;
    if (USE_GPU >= 0)
    {
        bool gpu = chooseGPU(USE_GPU);
        if (!gpu) return 0;
        cudaSetDevice(USE_GPU);
        cout << "Enabled GPU" << endl;
    }
    else
    {
        initializeGPU = false;
        cout << "Enabled CPU" << endl;
    }

    // Database setup (text)
    char h5name[256];
    sprintf(h5name, "ActivePolarNematicVertex_c%.3f_e%.3f.h5", zeta_c, zeta_e);
    simpleVertexDatabase h5db(numpts * 2, h5name, fileMode::replace);

    // Approximate Nvertices for hexagonal lattice
    int approx_Nvertices =numpts*2;


    // Create the model
    shared_ptr<VertexQuadraticEnergyWithNematic> model = make_shared<VertexQuadraticEnergyWithNematic>(numpts, a0, p0, reproducible, false,initializeGPU);

    // Set up regular hexagonal lattice for cell positions
    int nx = static_cast<int>(round(sqrt(static_cast<double>(numpts))));
    int ny = numpts / nx; // assumes divides evenly
    double d = sqrt(2.0 / sqrt(3.0));
    double Lx = static_cast<double>(nx) * d;
    double Ly = static_cast<double>(ny) * (sqrt(3.0) / 2.0) * d;
    model->Box->setGeneral(Lx, 0.0, 0.0, Ly);

    {
        ArrayHandle<double2> h_p(model->cellPositions, access_location::host, access_mode::overwrite);
        int idx = 0;
        for (int jj = 0; jj < ny; ++jj) {
            double y = static_cast<double>(jj) * (sqrt(3.0) / 2.0 * d);
            double xoff = (jj % 2 == 0 ? 0.0 : 0.5 * d);
            for (int ii = 0; ii < nx; ++ii) {
                double x = static_cast<double>(ii) * d + xoff;
                h_p.data[idx] = make_double2(x, y);
                model->Box->putInBoxReal(h_p.data[idx]);
                ++idx;
            }
        }
    }

    model->setCellsVoronoiTesselation(false);

    // Set uniform moduli (KA default 1.0, KP from param)
    model->setModuliUniform(1.0, kp);
    model->setUseNematic(true);

    // Initial cell types all 0 (passive)
    vector<int> types(numpts, 0);
    // Set T1 setT1Threshold
    model->setT1Threshold(0.01);
    // Create updater
    shared_ptr<selfPropelledCellVertexDynamics> updater = make_shared<selfPropelledCellVertexDynamics>(numpts, approx_Nvertices);

    // Set nematic activity
    vector<double> zetas(2, zeta_e); // Fix: size to number of types (0 and 1), passive for type 0, active for type 1
    zetas[1] = zeta_c;
    model->setActivity(zetas);

    // Set non-uniform motility (v0 for central, Dr uniform)
    vector<double> v0s(numpts, alpha_e);
    vector<double> drs(numpts, Dr);

    // Randomize cell types
    {
        std::vector<int> indices(numpts);
        std::iota(indices.begin(), indices.end(), 0);
        std::default_random_engine generator(42);
        std::shuffle(indices.begin(), indices.end(), generator);
        for (int k = 0; k < cluster_size; ++k) {
            int idx = indices[k];
            v0s[idx] = alpha_c;
            types[idx] = 1;  // Type 1 for orange
        }
    }
    model->setCellType(types);
    model->setCellPreferencesUniform(1.0,p0);
    model->setCellMotility(v0s, drs); 
    // Simulation setup
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(model);
    sim->addUpdater(updater, model);
    sim->setIntegrationTimestep(dt);
    sim->setCPUOperation(!initializeGPU);
    sim->setReproducible(reproducible);

    // Run initialization (all passive)
    printf("starting initialization\n");
    for (int ii = 0; ii < initSteps; ++ii)
    {
        sim->performTimestep();
    }
    printf("Finished with initialization\n");

    // Disrupt symmetry with small random displacements to vertices
    GPUArray<double2> displacements;
    displacements.resize(model->getNvertices());
    int Nv = model->getNvertices();
    vector<double2> rand_disp(Nv);
    std::default_random_engine generator(42);
    std::uniform_real_distribution<double> distribution(-0.001, 0.001);
    for (int i = 0; i < Nv; ++i) {
        rand_disp[i].x = distribution(generator);
        rand_disp[i].y = distribution(generator);
    }
    fillGPUArrayWithVector(rand_disp, displacements);
    model->moveDegreesOfFreedom(displacements);

    // Run production
    t1 = clock();
    for (int ii = 0; ii < tSteps; ++ii)
    {
        if(ii%100==0)
        {
        // model->reportMeanCellForce(true);
        // model->reportMeanVertexForce(true);
          h5db.writeState(model);
          printf("timestep: %d zeta_c: %.3f zeta_e: %.3f\n", ii, zeta_c, zeta_e);
        };
        sim->performTimestep();
    }
    t2 = clock();

    double steptime = (t2 - t1) / (double)CLOCKS_PER_SEC;
    printf("simulation took: ~ %f s\n", steptime);

    if (initializeGPU)
        cudaDeviceReset();
    return 0;
}
