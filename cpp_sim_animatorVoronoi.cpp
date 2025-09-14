#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#include "Simulation.h"
#include "databaseTextVoronoi.h"
#include "periodicBoundaries.h"
#include "voronoiQuadraticEnergy.h"  // Add this for the concrete class
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

int main(int argc, char*argv[])
{
    // default parameters same as simulation
    int numpts = 400;
    int USE_GPU = 0; // ignored
    int c;
    int tSteps = 262144;
    int initSteps = 1; // ignored

    double dt = 0.01;
    double p0 = 3.8;
    double a0 = 1.0; // ignored
    double v0 = 0.1; // ignored
    double Dr = 0.5; // ignored
    double J = 0.0;
    int S = 3; // ignored
    int fIdx = 0; // ignored
    int fps = 30; 

    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:j:x:y:z:p:f:t:e:s:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'j': J = atof(optarg); break;
            case 'f': fps = atoi(optarg); break; 
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

    bool reproducible = false; // ignored, but set to false

    // construct input filename
    char txtname[256];
    sprintf(txtname, "aligningVoronoi_modified_%.3f_%.3f.txt", J, p0);

    // output video name
    char viname[256];
    sprintf(viname, "../visualizationTools/aligningVoronoi_modified_%.3f_%.3f.avi", J, p0);

    // open the text file
    std::ifstream inputFile(txtname);
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open " << txtname << std::endl;
        return 1;
    }

    // assume square box with side length sqrt(N), since total area ~ N * a0 = N
    double L = sqrt(static_cast<double>(numpts));
    shared_ptr<periodicBoundaries> Box = make_shared<periodicBoundaries>(L, 0.0, 0.0, L);

    // setup rendering
    const int ws = 800;
    double scale = ws / L;

    // setup video writer
    cv::VideoWriter writer(viname, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(ws, ws));

    int maxNeigh = 32; // from class

    // data structures
    std::vector<double2> positions(numpts);
    std::vector<int> neighborNums(numpts);
    std::vector<double2> voroCur(numpts * maxNeigh);
    std::vector<int> cellTypes(numpts);

    // variables for cluster
    int centerTag = -1;
    std::vector<int> cluster;
    std::vector<cv::Scalar> clusterColors = {
        cv::Scalar(0, 0, 255),    // Red
        cv::Scalar(255, 0, 0),    // Blue
        cv::Scalar(255, 0, 255),  // Magenta
        cv::Scalar(255, 255, 0),  // Cyan
        cv::Scalar(0, 255, 0)     // Green
    };
    std::vector<cv::Scalar> fillColors(numpts, cv::Scalar(-1, -1, -1));

    bool firstFrame = true;
    unsigned long frameCount = 0;

    std::string line;
    while (std::getline(inputFile, line)) {
        // parse time line "t=<time>"
        double time;
        if (sscanf(line.c_str(), "t=%lf", &time) != 1) {
            if (line.empty()) continue; // skip empty lines if any
            std::cerr << "Invalid time line: " << line << std::endl;
            break;
        }

        // read and parse cell lines
        for (int ii = 0; ii < numpts; ++ii) {
            if (!std::getline(inputFile, line)) {
                std::cerr << "Unexpected end of file during frame " << frameCount << std::endl;
                return 1;
            }

            // parse cell line
            size_t cell_start = line.find("cell[");
            if (cell_start == std::string::npos) {
                std::cerr << "Invalid cell line: " << line << std::endl;
                return 1;
            }

            size_t bracket_close = line.find("]", cell_start + 5);
            if (bracket_close == std::string::npos) {
                std::cerr << "No closing bracket in line: " << line << std::endl;
                return 1;
            }

            std::string type_str = line.substr(cell_start + 5, bracket_close - cell_start - 5);
            int cell_type;
            try {
                cell_type = std::stoi(type_str);
            } catch (...) {
                std::cerr << "Invalid cell type in line: " << line << std::endl;
                return 1;
            }

            size_t eq_pos = line.find("=", bracket_close + 1);
            if (eq_pos == std::string::npos || eq_pos != bracket_close + 1) {
                std::cerr << "No equals sign after bracket in line: " << line << std::endl;
                return 1;
            }

            std::string content = line.substr(eq_pos + 1);

            size_t vertices_pos = content.find("vertices=[");
            if (vertices_pos == std::string::npos) {
                std::cerr << "Invalid vertices in line: " << line << std::endl;
                return 1;
            }

            // parse position
            std::string pos_str = content.substr(0, vertices_pos);
            std::replace(pos_str.begin(), pos_str.end(), ',', ' ');
            std::stringstream pos_ss(pos_str);
            double x, y;
            pos_ss >> x >> y;

            positions[ii] = make_double2(x, y);
            cellTypes[ii] = cell_type;

            // parse vertices
            size_t open_pos = vertices_pos + 10;
            size_t close_pos = content.find("]", open_pos);
            if (close_pos == std::string::npos) {
                std::cerr << "No closing ] in line: " << line << std::endl;
                return 1;
            }

            std::string verts_line = content.substr(open_pos, close_pos - open_pos);
            std::replace(verts_line.begin(), verts_line.end(), ',', ' ');
            std::stringstream verts_ss(verts_line);

            std::vector<double> vals;
            double val;
            while (verts_ss >> val) {
                vals.push_back(val);
            }

            if (vals.size() % 2 != 0) {
                std::cerr << "Odd number of vertex coordinates in line: " << line << std::endl;
                return 1;
            }

            int neighs = vals.size() / 2;
            if (neighs > maxNeigh) {
                std::cerr << "Too many neighbors (" << neighs << ") in line: " << line << std::endl;
                return 1;
            }

            neighborNums[ii] = neighs;
            for (int nn = 0; nn < neighs; ++nn) {
                double vx = vals[2 * nn];
                double vy = vals[2 * nn + 1];
                double vrelx = vx - x;
                double vrely = vy - y;
                voroCur[maxNeigh * ii + nn] = make_double2(vrelx, vrely);
            }
        }

        if (firstFrame) {
            // find central cluster
            double2 center = make_double2(L / 2.0, L / 2.0);
            double minD = 1e10;
            for (int tag = 0; tag < numpts; ++tag) {
                double2 pos = positions[tag];
                double2 diff;
                Box->minDist(pos, center, diff);
                double d = norm(diff);
                if (d < minD) {
                    minD = d;
                    centerTag = tag;
                }
            }

            // find 4 closest to centerTag
            double2 centerPos = positions[centerTag];
            std::vector<std::pair<double, int>> dists;
            for (int tag = 0; tag < numpts; ++tag) {
                if (tag == centerTag) continue;
                double2 pos = positions[tag];
                double2 diff;
                Box->minDist(pos, centerPos, diff);
                double d = norm(diff);
                dists.emplace_back(d, tag);
            }
            std::sort(dists.begin(), dists.end());

            cluster = {centerTag};
            for (int k = 0; k < 4; ++k) {
                cluster.push_back(dists[k].second);
            }

            firstFrame = false;
        }

        // set fill colors based on cell types
        for (int tag = 0; tag < numpts; ++tag) {
            fillColors[tag] = clusterColors[cellTypes[tag] % clusterColors.size()];
        }

        // modify shading for central cluster (lighter shade)
        for (int clus_tag : cluster) {
            fillColors[clus_tag] = (fillColors[clus_tag] + cv::Scalar(255, 255, 255)) * 0.5;
        }

        // extended image
        cv::Mat ext_img(3 * ws, 3 * ws, CV_8UC3, cv::Scalar(255, 255, 255));

        // draw cells
        for (int tag = 0; tag < numpts; ++tag) {
            int idx = tag; // assume tagToIdx is identity
            int neighs = neighborNums[idx];
            if (neighs < 3) continue;

            double2 cpos = positions[idx];

            std::vector<cv::Point> base_pts(neighs);
            for (int nn = 0; nn < neighs; ++nn) {
                double2 vrel = voroCur[maxNeigh * idx + nn];
                double2 vabs = make_double2(cpos.x + vrel.x, cpos.y + vrel.y);
                base_pts[nn] = cv::Point(static_cast<int>(vabs.x * scale), static_cast<int>(vabs.y * scale));
            }

            cv::Scalar fillColor = fillColors[tag];

            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    std::vector<cv::Point> shifted_pts(neighs);
                    for (int nn = 0; nn < neighs; ++nn) {
                        shifted_pts[nn] = base_pts[nn] + cv::Point(i * ws, j * ws);
                    }

                    if (fillColor.val[0] >= 0) {
                        cv::fillPoly(ext_img, {shifted_pts}, fillColor);
                    }
                    cv::polylines(ext_img, {shifted_pts}, true, cv::Scalar(0, 0, 0), 1);
                }
            }
        }

        // draw centers
        for (int tag = 0; tag < numpts; ++tag) {
            double2 pos = positions[tag];
            int px = static_cast<int>(pos.x * scale);
            int py = static_cast<int>(pos.y * scale);

            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    cv::circle(ext_img, cv::Point(px + i * ws, py + j * ws), 2, cv::Scalar(0, 0, 0), -1);
                }
            }
        }

        // extract central part
        cv::Mat final_img = ext_img(cv::Rect(ws, ws, ws, ws)).clone();

        // write to video
        writer << final_img;

        frameCount++;
        if (frameCount >= tSteps) {
            break;
        }
    }

    writer.release();

    return 0;
};
