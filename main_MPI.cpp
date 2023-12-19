#include <dirent.h>
#include <mpi.h>

#include <alps/alea/batch.hpp>
#include <alps/utilities/mpi.hpp>
#include <argparse.hpp>
#include <automodel.hpp>
#include <autoobservable.hpp>
#include <exec_parallel.hpp>
#include <filesystem>
#include <funcs.hpp>
#include <functional>
#include <iostream>
#include <jackknife.hpp>
#include <libconfig.h++>
#include <observable.hpp>
#include <string>

#include "MainConfig.h"

namespace fs = std::filesystem;
fs::path project_dir = fs::path(PROJECT_DIR);

fs::path out_dir_name = fs::path("output_worm");

int main(int argc, char **argv) {
  int rank;
  int size;
  double elapsed;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // set up alps::mpi::reducer
  alps::alea::mpi_reducer red_(alps::mpi::communicator(), 0);
  alps::alea::reducer_setup setup = red_.get_setup();

  char tmp[256];
  auto *_ = getcwd(tmp, 256);

  Config cfg;
  cfg.setAutoConvert(true);

  //* argparse
  argparse::ArgumentParser parser("test", "argparse test program",
                                  "Apache License 2.0");

  parser.addArgument({"-L1"}, "set shape[0]");
  parser.addArgument({"--sps"}, "set spin degree of freedom");
  parser.addArgument({"-L2"}, "set shape[1]");
  parser.addArgument({"-L3"}, "set shape[2]");
  parser.addArgument({"-N"}, "# of montecarlo steps (sweeps)");
  parser.addArgument({"-K"}, "# of montecarlo steps for thermalization");
  parser.addArgument(
      {"--split-sweeps"},
      "bool that determines whether to split # sweeps among processes",
      argparse::ArgumentType::StoreTrue);
  parser.addArgument({"--z"}, "bool : introduce zero worm",
                     argparse::ArgumentType::StoreTrue);

  parser.addArgument({"-T"}, "set temperature");
  parser.addArgument({"-m"}, "model name");
  parser.addArgument({"-ham"}, "path to hamiltonian");
  parser.addArgument({"-unitary"}, "path to unitary");
  parser.addArgument({"-obs"}, "path to observables");
  parser.addArgument({"-wobs"}, "path to worm observables");
  parser.addArgument({"-P1"}, "set params[0]");
  parser.addArgument({"-P2"}, "set params[1]");
  parser.addArgument({"-alpha"}, "set alpha");
  parser.addArgument({"--output"}, "save output to file",
                     argparse::ArgumentType::StoreTrue);

  auto args = parser.parseArgs(argc, argv);

  if (rank == 0) {
    std::cout << "current working directory : " << tmp << std::endl;
  }
  try {
    cfg.readFile("../config/model.cfg");
  } catch (const FileIOException &fioex) {
    cerr << "I/O error while reading file." << std::endl;
    return (EXIT_FAILURE);
  } catch (const ParseException &px) {
    cerr << "error while parsing items" << std::endl;
    cerr << "Maybe some list include multiple types (e.g. = [1.0, 1, 1])"
         << std::endl;
    return (EXIT_FAILURE);
  }

  const Setting &root = cfg.getRoot();
  string model_name = root["model"];
  bool print_lat = (bool)root["print_lattice"];
  model_name = args.safeGet<string>("m", model_name);

  const Setting *mcfg;
  try {
    mcfg = &root["models"][model_name];
  } catch (const SettingNotFoundException &nfex) {
    throw std::runtime_error("model name not found");
    return (EXIT_FAILURE);
  }
  // const Setting& mcfg = root["models"][model_name];
  const Setting &shape_cfg = mcfg->lookup("length");
  const Setting &params_cfg = mcfg->lookup("params");
  const Setting &types_cfg = mcfg->lookup("types");
  const Setting &dofs_cfg = mcfg->lookup("dofs");

  double shift;
  size_t ns_unit;
  string file;
  string basis;
  string cell;
  string ham_path;
  string obs_path;
  string u_path;
  fs::path out_file;
  fs::path output_folder;
  vector<string> wobs_paths;
  vector<size_t> shapes;
  vector<int> types;
  vector<double> params;
  vector<size_t> dofs;
  bool repeat;  // true if repeat params and types.
  bool zero_worm;

  for (int i = 0; i < shape_cfg.getLength(); i++) {
    int tmp = shape_cfg[i];
    shapes.push_back(tmp);
  }
  for (int i = 0; i < dofs_cfg.getLength(); i++) {
    dofs.push_back((size_t)dofs_cfg[i]);
  }
  for (int i = 0; i < params_cfg.getLength(); i++) {
    params.push_back((float)params_cfg[i]);
  }
  for (int i = 0; i < types_cfg.getLength(); i++) {
    types.push_back(types_cfg[i]);
  }

  file = string(mcfg->lookup("file").c_str());
  basis = string(mcfg->lookup("basis").c_str());
  cell = string(mcfg->lookup("cell").c_str());
  ham_path = string(mcfg->lookup("ham_path").c_str());
  try {
    u_path = string(mcfg->lookup("u_path").c_str());
  } catch (const SettingNotFoundException &nfex) {
    u_path = "";
  }
  try {
    obs_path = string(mcfg->lookup("obs_path").c_str());
  } catch (const SettingNotFoundException &nfex) {
    obs_path = "";
  }
  repeat = (bool)mcfg->lookup("repeat");
  shift = (double)mcfg->lookup("shift");
  double alpha = (double)mcfg->lookup("alpha");
  zero_worm = (bool)mcfg->lookup("zero_worm");

  //* settings for monte-carlo
  const Setting &settings = root["mc_settings"];

  size_t sweeps;
  size_t therms;
  size_t cutoff_l;
  size_t seed;
  size_t sps;
  double T = 0;
  bool fix_wdensity = false;
  try {
    const Setting &config = settings["config"];
    sweeps = (long)config.lookup("sweeps");
    therms = (long)config.lookup("therms");
    cutoff_l = (long)config.lookup("cutoff_length");
    T = (double)config.lookup("temperature");
    fix_wdensity = config.lookup("fix_wdensity");
    seed = static_cast<size_t>(config.lookup("seed"));

  } catch (...) {
    std::cout << "I/O error while reading mc_settings.default settings"
              << std::endl;
    std::cout << "read config file from default instead" << std::endl;
    const Setting &config = settings["default"];
    sweeps = (long)config.lookup("sweeps");
    therms = (long)config.lookup("therms");
    cutoff_l = (long)config.lookup("cutoff_length");
    T = (double)config.lookup("temperature");
    fix_wdensity = config.lookup("fix_wdensity");
    seed = static_cast<size_t>(config.lookup("seed"));
  }

  try {
    ham_path = args.get<string>("ham");
    try {
      u_path = args.get<string>("unitary");
    } catch (...) {
      u_path = "";
    }
    try {
      obs_path = args.get<string>("obs");
    } catch (...) {
      if (rank == 0)
        std::cout
            << "obs_path is not given. Elements of observables are set to zero"
            << std::endl;
      obs_path = "";
    }
    try {
      wobs_paths = vector<string>(1, args.get<string>("wobs"));
    } catch (...) {
      if (rank == 0)
        std::cout
            << "wobs_path is not given. Elements of worm observables will set "
               "to zero"
            << std::endl;
    }
  } catch (...) {
    // std::cout << "I/O error while reading mc_settings.default settings" <<
    // std::endl;
  }

  // parser
  sps = args.safeGet<size_t>("sps", dofs[0]);
  std::fill(dofs.begin(), dofs.end(), sps);
  shapes[0] = args.safeGet<size_t>("L1", shapes[0]);
  shapes[1] = args.safeGet<size_t>("L2", shapes[1]);
  shapes[2] = args.safeGet<size_t>("L3", shapes[2]);
  T = args.safeGet<double>("T", T);
  sweeps = args.safeGet<int>("N", sweeps);
  therms = args.safeGet<int>("K", therms);
  params[0] = args.safeGet<float>("P1", params[0]);
  params[1] = args.safeGet<float>("P2", params[1]);
  alpha = args.safeGet<double>("alpha", alpha);

  if (args.has("split-sweeps")) {
    sweeps = sweeps / size;
  }
  sweeps = (sweeps / 2) * 2;  // make sure sweeps is even number

  if (args.has("output") && rank == 0) {
    fs::path base_output_folder = project_dir / out_dir_name;
    string shape;
    if (shapes.size() == 1) {
      shape += ("L_" + to_string(shapes[0]));
    } else if (shapes.size() == 2) {
      shape += ("L_" + to_string(shapes[0]) + "_" + to_string(shapes[1]));
    } else if (shapes.size() == 3) {
      shape += ("L_" + to_string(shapes[0]) + "_" + to_string(shapes[1]) + "_" +
                to_string(shapes[2]));
    }
    string setting_name =
        "T_" + to_string(T) + "/" + "N_" + to_string(sweeps * size);
    // output_folder = folder + model_name + "/" + shape + "/" + setting_name;
    output_folder = base_output_folder / model_name / shape / setting_name;

    hash<std::string> hasher;
    size_t hash = hasher(ham_path);

    // out_file = output_folder + "/" + getCurrentDateTime() + "_" +
    // to_string(hash) + ".txt";
    out_file =
        output_folder / (getCurrentDateTime() + "_" + to_string(hash) + ".txt");

    if (fs::create_directories(out_file.parent_path())) {
      std::cout << "create directory : " << out_file.parent_path() << std::endl;
    }
  }

  try {
    ns_unit = (size_t)mcfg->lookup("ns_unit");
  } catch (...) {
    if (rank == 0) {
      std::cout << "Warning : please set ns_unit in model.cfg" << std::endl;
      std::cout << "Warning : Ns_unit was not given. Automatically set to 1"
                << std::endl;
    }
    ns_unit = 1;
  }
  try {
    const Setting &wobs_path_list = mcfg->lookup("worm_obs_path");
    for (int i = 0; i < wobs_path_list.getLength(); i++) {
      string tmp = wobs_path_list[i];
      wobs_paths.push_back(tmp);
    }
  } catch (const SettingNotFoundException &nfex) {
    if (rank == 0)
      std::cout << "Warning : No worm observables was given" << std::endl;
  }

  if (args.has("z")) zero_worm = true;

  if (rank == 0) {
    if (!out_file.empty()) {
      std::cout << "The result will be written in : " << fs::absolute(out_file)
                << std::endl;
      freopen(out_file.c_str(), "w", stdout);
    }
  }

  if (rank == 0) {
    std::cout << "model name is : " << model_name << std::endl;
    std::cout << "run on  : " << size << " nodes" << std::endl;
    std::cout << "zero_wom : " << (zero_worm ? "YES" : "NO") << std::endl;
    std::cout << "repeat : " << (repeat ? "YES" : "NO") << std::endl;
    std::cout << "params : " << params << std::endl;
    std::cout << "alpha : " << alpha << std::endl;
    std::cout << "temperature : " << T << std::endl;
  }

  //* finish argparse
  model::base_lattice lat(basis, cell, shapes, file, !rank);
  // model::base_model<bcl::heatbath>* spin;
  std::unique_ptr<model::base_model<bcl::heatbath>> spin_ptr;
  if (u_path.empty()) {
    if (rank == 0)
      std::cout << "unitary is not given. Identity matrix is used."
                << std::endl;
    spin_ptr = std::make_unique<model::base_model<bcl::heatbath>>(
        lat, dofs, ham_path, params, types, shift, zero_worm, repeat, !rank,
        alpha);
  } else {
    spin_ptr = std::make_unique<model::base_model<bcl::heatbath>>(
        lat, dofs, ham_path, u_path, params, types, shift, zero_worm, repeat,
        !rank, alpha);
  }
  model::observable obs(*spin_ptr, obs_path, !rank);

  // n* set wobs
  if (wobs_paths.size() == 0) wobs_paths.push_back("");
  model::MapWormObs mapwobs;
  for (int i = 0; i < wobs_paths.size(); i++) {
    string name = "G";
    name += to_string(i);
    mapwobs.push_back(
        name, model::WormObs(spin_ptr->sps_sites(0), wobs_paths[i], !rank));
  }

  // model::WormObs wobs(spin.sps_sites(0), wobs_path, !rank); // all elements
  // of sps_sites are the same.

  size_t n_sites = lat.L * ns_unit;

  // output MC step info
  if (rank == 0)
    std::cout << "----------------------------------------" << std::endl
              << "therms(each process)    : " << therms << std::endl
              << "sweeps(each process)    : " << sweeps << std::endl
              << "sweeps(in total)        : " << sweeps * size << std::endl;

  if (rank == 0) {
    for (int i = 0; i < 40; i++) std::cout << "-";
    std::cout << std::endl;
  }

  // n* check argument
  if (T < 0) {
    throw std::runtime_error("temperature must be positive");
  }

  alps::alea::autocorr_result<double> ac_res;

  // simulate with worm algorithm (parallel computing is enable)
  vector<batch_res> res;
  auto map_worm_obs =
      exe_worm_parallel(*spin_ptr, T, sweeps, therms, cutoff_l, fix_wdensity,
                        rank, res, ac_res, obs, mapwobs, seed);

  batch_res as = res[0];   // average sign
  batch_res ene = res[1];  // signed energy i.e. $\sum_i E_i S_i / N_MC$
  batch_res n_neg_ele = res[2];
  batch_res n_ops = res[3];
  batch_res N2 = res[4];
  batch_res N = res[5];
  batch_res dH = res[6];   // $\frac{\frac{\partial}{\partial h}Z}{Z}$
  batch_res dH2 = res[7];  // $\frac{\frac{\partial^2}{\partial h^2}Z}{Z}$
  batch_res phys_conf = res[8];

  vector<pair<string, batch_res>> worm_obs;
  int i = 0;
  for (auto &obs : map_worm_obs) {
    worm_obs.emplace_back(obs.first, res[9 + i]);
    i++;
  }

  as.reduce(red_);
  ene.reduce(red_);
  n_neg_ele.reduce(red_);
  n_ops.reduce(red_);
  N2.reduce(red_);
  N.reduce(red_);
  dH.reduce(red_);
  dH2.reduce(red_);
  phys_conf.reduce(red_);
  ac_res.reduce(red_);

  for (auto &obs : worm_obs) {
    get<1>(obs).reduce(red_);
  }

  double elapsed_max;
  double elapsed_min;
  MPI_Allreduce(&elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&elapsed, &elapsed_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  if (setup.have_result) {
    std::function<double(double, double, double)> f;

    pair<double, double> as_mean =
        jackknife_reweight_single(as);  // calculate <S>
    pair<double, double> nop_mean =
        jackknife_reweight_single(n_ops);  // calculate <S>
    pair<double, double> nnop_mean =
        jackknife_reweight_single(n_neg_ele);  // calculate <S>

    // n* install
    pair<double, double> ene_mean =
        jackknife_reweight_div(ene, as);  // calculate <SH> / <S>

    // calculate worm_observable
    vector<pair<string, pair<double, double>>> worm_obs_mean;
    for (auto &obs : worm_obs) {
      auto mean = jackknife_reweight_div(get<1>(obs),
                                         phys_conf);  // calculate <WoS> / <S>
      worm_obs_mean.emplace_back(obs.first, mean);
    }

    // calculat heat capacity
    f = [](double x1, double x2, double y) {
      return (x2 - x1) / y - (x1 / y) * (x1 / y);
    };
    pair<double, double> c_mean = jackknife_reweight_any(N, N2, as, f);

    // calculate magnetization
    pair<double, double> m_mean = jackknife_reweight_div(dH, as);

    // calculate susceptibility
    f = [](double x1, double x2, double y) {
      return x2 / y - (x1 / y) * (x1 / y);
    };
    pair<double, double> chi_mean = jackknife_reweight_any(dH, dH2, as, f);

    std::cout << "Elapsed time         = " << elapsed_max << "(" << elapsed_min
              << ") sec\n"
              << "Speed                = " << (therms + sweeps) / elapsed_max
              << " MCS/sec\n";

    std::cout << "beta                 = " << 1.0 / T << std::endl
              << "Total Energy         = " << ene_mean.first << " +- "
              << ene_mean.second << std::endl;

    std::cout << "Average sign         = " << as_mean.first << " +- "
              << as_mean.second << std::endl
              << "Energy per site      = " << ene_mean.first / n_sites << " +- "
              << ene_mean.second / n_sites << std::endl
              << "Specific heat        = " << c_mean.first / n_sites << " +- "
              << c_mean.second / n_sites << std::endl
              << "magnetization        = " << m_mean.first * T / n_sites
              << " +- " << m_mean.second * T / n_sites << std::endl
              << "susceptibility       = " << chi_mean.first * T / n_sites
              << " +- " << chi_mean.second * T / n_sites << std::endl;

    for (auto &obs : worm_obs_mean) {
      fillStringWithSpaces(obs.first, 11);
      std::cout << obs.first << "          = " << obs.second.first << " +- "
                << obs.second.second << std::endl;
    }

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Integrated correlation time " << std::endl
              << "H                    = " << ac_res.tau()[0] << std::endl
              << "M^2                  = " << ac_res.tau()[1] << std::endl
              << "S                    = " << ac_res.tau()[2] << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "# of operators       = " << nop_mean.first << " +- "
              << nop_mean.second << std::endl
              << "# of neg sign op     = " << nnop_mean.first << " +- "
              << nnop_mean.second << std::endl;
  }
  MPI_Finalize();
}
