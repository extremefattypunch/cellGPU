#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#include "Simulation.h"
#include "voronoiQuadraticEnergy.h"
#include "selfPropelledAligningParticleDynamics.h"
#include "simpleVoronoiDatabase.h"
#include "databaseTextVoronoi.h"

int main(int argc, char*argv[])
{
    //...some default parameters
    int numpts = 400; //number of cells
    int USE_GPU = 0; //0 or greater uses a gpu, any negative number runs on the cpu
    int c;
    int tSteps = 262144; //number of time steps to run after initialization
    int initSteps = 1; //number of initialization steps

    double dt = 0.01; //the time step size
    double p0 = 3.8;  //the preferred perimeter
    double a0 = 1.0;  // the preferred area
    double v0 = 0.1;  // the self-propulsion
    double Dr  =0.5;  // rotational diffusion
    double J = 0.0;   //alignment coupling
    int S = 1;   //0 for none 1 for analysis only 2 for animation only 3 for both
    int fIdx = 0;
    //The defaults can be overridden from the command line
    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:j:x:y:z:p:f:t:e:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'j': J = atof(optarg); break;
            case 'f': fIdx = atoi(optarg); break; 
            case 'e': dt = atof(optarg); break;
            case 'p': p0 = atof(optarg); break;
            case 'a': a0 = atof(optarg); break;
            case 'v': v0 = atof(optarg); break;
            case 's': S = atof(optarg); break;
            case '?':
                    if(optopt=='c')
                        std::cerr<<"Option -" << optopt << "requires an argument.\n";
                    else if(isprint(optopt))
                        std::cerr<<"Unknown option '-" << optopt << "'.\n";
                    else
                        std::cerr << "Unknown option character.\n";
                    return 1;
            default:
                       abort();
        };

    clock_t t1,t2; //clocks for timing information
    bool reproducible = false; // if you want random numbers with a more random seed each run, set this to false
    //check to see if we should run on a GPU
    bool initializeGPU = true;
    if (USE_GPU >= 0)
        {
        bool gpu = chooseGPU(USE_GPU);
        if (!gpu) return 0;
        cudaSetDevice(USE_GPU);
        cout << "enabled GPU" << endl;
        }
    else
        {
        initializeGPU = false;
        cout << "enabled CPU" << endl;
        }

    char h5name[256];
    sprintf(h5name,"aligningVoronoi_modified_%.3f_%.3f.h5",J,p0);
    simpleVoronoiDatabase h5db(numpts, h5name, fileMode::replace);

    char txtname[256];
    sprintf(txtname,"aligningVoronoi_modified_%.3f_%.3f.txt",J,p0);
    DatabaseTextVoronoi txtdb(txtname, fileMode::replace);

    //define an equation of motion object...here for self-propelled cells
    shared_ptr<selfPropelledAligningParticleDynamics> spp = make_shared<selfPropelledAligningParticleDynamics>(numpts);
    spp->setJ(J);
    //define a voronoi configuration with a quadratic energy functional
    shared_ptr<VoronoiQuadraticEnergy> spv  = make_shared<VoronoiQuadraticEnergy>(numpts,1.0,4.0,reproducible);

    //set the cell preferences to uniformly have A_0 = 1, P_0 = p_0
    spv->setCellPreferencesUniform(1.0,p0);
    //set the cell activity to have D_r = 1. and a given v_0
    spv->setv0Dr(v0,Dr);
  //set uniform cell type

    spv->setCellTypeUniform(0);

    //combine the equation of motion and the cell configuration in a "Simulation"
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(spv);
    sim->addUpdater(spp,spv);
    //set the time step size
    sim->setIntegrationTimestep(dt);
    //initialize Hilbert-curve sorting... can be turned off by commenting out this line or seting the argument to a negative number
    //sim->setSortPeriod(initSteps/10);
    //set appropriate CPU and GPU flags
    sim->setCPUOperation(!initializeGPU);
    sim->setReproducible(reproducible);

    //run for a few initialization timesteps
    printf("starting initialization\n");
    for(int ii = 0; ii < initSteps; ++ii)
        {
        sim->performTimestep();
        };
    printf("Finished with initialization\n");
    // cout << "current q = " << spv->reportq() << endl;
    //the reporting of the force should yield a number that is numerically close to zero.
    spv->reportMeanCellForce(false);

    //run for additional timesteps, and record timing information
    t1=clock();
    for(int ii = 0; ii < tSteps; ++ii)
        {

        if((ii>tSteps/2) && (ii%10==0))
            {
            double2 vPar, vPerp;
            double val = spv->vicsekOrderParameter(vPar, vPerp);
            printf("J:%f, p0:%f\t\t timestep %i\t\t energy %f\t\t phi %f \n", J, p0, ii, spv->computeEnergy(), val);
            if(S==1 || S==3){
            h5db.writeState(spv);
            }
            if(S==2 || S==3)
            txtdb.writeState(spv);
            };
        sim->performTimestep();
        };
    t2=clock();

    double steptime = (t2-t1)/(double)CLOCKS_PER_SEC;
    // cout << "simulation took: ~ " << steptime << "s " << endl;
    // cout << "final meanShapeIndex: " << spv->reportq() << endl;

    if(initializeGPU)
        cudaDeviceReset();
    return 0;
};
