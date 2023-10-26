#include <automodel.hpp>
#include <vector>
#include <string>

#include <funcs.hpp>
#include <state.hpp>
#include <automodel.hpp>
#include <autoobservable.hpp>
#include <exec_parallel.hpp>
#include <options.hpp>
#include <argparse.hpp>
#include <observable.hpp>
#include <funcs.hpp>

#include "gtest/gtest.h"
#include "dataset.hpp"

using namespace std;

// TEST(HamsTest, SS_4x4_aggr) {
//   std::vector<size_t> shapes = {2, 2};
//   model::base_lattice lat("square lattice", "SS2", shapes, "../config/lattices.xml", true);
//   string ham_path = "../gtest/model_array/SS/H1";
//   model::base_model<bcl::st2013> spin(lat, {4, 4}, ham_path, {1,1}, {0,1}, 0.1, false, false, true);
//   cerr << lat.site_type << endl;
//   cerr << spin.num_types() << endl;
//   cerr << spin.nn_sites << endl;
//   cerr << spin.s_flip_max_weights << endl;
// }

typedef bcl::st2013 MC;
typedef std::exponential_distribution<> expdist_t;
typedef std::uniform_real_distribution<> uniform_t;
typedef spin_state::state_t state_t;
typedef spin_state::Operator OP_type;
uniform_t uniform;
// int seed = static_cast<unsigned>(time(0));
int seed = 1662533963;
auto rand_src = engine_type(seed);

TEST(SingleOptype, StateUpdate)
{
  size_t sps = 8;
  size_t max_size_t = std::numeric_limits<signed char>::max();
  cerr << "max_size_t = " << max_size_t << endl;
  std::vector<size_t> bonds = {1, 2, 3, 4, 5, 6}; // neighbor sites from 0
  std::vector<size_t> powers = {1, sps, sps * sps};
  state_t nn_state = {1, 4, 7, 0, 3, 6};
  size_t x1 = 4, x2 = 5;
  OP_type op = OP_type(&bonds, &powers, x1 + x2 * sps, nn_state, -1, 0);
  EXPECT_EQ(op._check_is_bond(), false);
  EXPECT_EQ(op.state(0), 4);
  EXPECT_EQ(op.state(1), 5);
  EXPECT_EQ(op.update_state(0, 5), 1 + 5 * sps);
  EXPECT_EQ(op.update_state(0, 3), 4 + 5 * sps);
  EXPECT_EQ(op.update_state(1, 3), 4 + 0 * sps);
  EXPECT_EQ(op.update_nn_state(0, 1)[0], 2);
  EXPECT_EQ(op.update_nn_state(2, 3)[2], 2);
}
