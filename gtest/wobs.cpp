#include <autoobservable.hpp>
#include <vector>
#include <funcs.hpp>

#include "dataset.hpp"
#include "gtest/gtest.h"

using namespace std;

TEST(BaseWormObs, GetState)
{
  model::BaseWormObs wo(4, 2); // dof, legs
  EXPECT_EQ(wo.GetState(1, 2, 3, 1), 121);
  EXPECT_EQ(wo.GetState(3, 2, 3, 1), 123);
  EXPECT_EQ(wo.GetState(0, 1, 0, 1), 68);
  model::BaseWormObs wo2(2, 2);
  EXPECT_EQ(wo2.GetState(1, 1, 0, 0), 3);
  EXPECT_EQ(wo2.GetState(1, 0, 0, 0), 1);
  model::BaseWormObs wo3(3, 2);
  EXPECT_EQ(wo3.GetState(1, 0, 1, 0), 10);
  EXPECT_EQ(wo3.GetState(1, 1, 1, 0), 13);
  model::BaseWormObs wo4(2, 1);
  EXPECT_EQ(wo4.GetState(1, 1), 3);
  EXPECT_EQ(wo4.GetState(1, 0), 1);
  model::BaseWormObs wo5(4, 1);
  EXPECT_EQ(wo5.GetState(1, 1), 5);
  EXPECT_EQ(wo5.GetState(1, 0), 1);
  EXPECT_EQ(wo5.GetState(1, 2), 9);
}


TEST(BaseWormObs, CallError)
{
  model::BaseWormObs wo(4, 1);
  // wo(1, 2, 3, 2);
  EXPECT_DEATH(double x = wo(1, 2),
               "BaseWormObs::operator is virtual function");
}

TEST(BaseWormObs, ErrorCase)
{
  model::BaseWormObs wo(2, 1);
  EXPECT_DEATH(wo.GetState(1, 2), "spin state is out of range");
  EXPECT_DEATH(model::BaseWormObs(1, 3), "leg size of BaseWormObs must be 1 or 2");
}

TEST(ArrWormObs, GetState)
{
  model::ArrWormObs wo(4, 2); // dof, legs
  EXPECT_EQ(wo.GetState(1, 2, 3, 1), 121);
  EXPECT_EQ(wo.GetState(3, 2, 3, 1), 123);
  EXPECT_EQ(wo.GetState(0, 1, 0, 1), 68);
  model::ArrWormObs wo2(2, 2);
  EXPECT_EQ(wo2.GetState(1, 1, 0, 0), 3);
  EXPECT_EQ(wo2.GetState(1, 0, 0, 0), 1);
  model::ArrWormObs wo3(3, 2);
  EXPECT_EQ(wo3.GetState(1, 0, 1, 0), 10);
  EXPECT_EQ(wo3.GetState(1, 1, 1, 0), 13);
  model::ArrWormObs wo4(2, 1);
  EXPECT_EQ(wo4.GetState(1, 1), 3);
  EXPECT_EQ(wo4.GetState(1, 0), 1);
  model::ArrWormObs wo5(4, 1);
  EXPECT_EQ(wo5.GetState(1, 1), 5);
  EXPECT_EQ(wo5.GetState(1, 3), 13);
  EXPECT_EQ(wo5.GetState(1, 2), 9);
}

TEST(ArrWormObs, SetVector)
{
  model::ArrWormObs wo(4, 2); // dof, legs

  std::vector<double> v;
  for (int i = 0; i < std::pow(4, 4); i++)
  {
    v.push_back(i);
  }
  wo._SetWormObs(v);
  EXPECT_EQ(wo(1, 2, 3, 1), 121);
  testing::internal::CaptureStdout();
  wo._SetWormObs(v);
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_STREQ("Worm_obs is already set\n", output.c_str());
}

TEST(ArrWormObs, CheckReadError)
{
  model::ArrWormObs wo(4, 2);                              // dof, legs
  EXPECT_DEATH(wo.ReadNpy("test.np"), "File path is not"); // expected message is File path is not *.npy
  EXPECT_DEATH(wo.ReadNpy("_test.npy"), "File does not exist");
  model::ArrWormObs wo2(4, 2);
  EXPECT_DEATH(wo2.ReadNpy("../gtest/model_array/worm_obs/test.npy"), "Require 2D array"); // 1D array
  // EXPECT_DEATH(wo2.ReadNpy("../gtest/model_array/worm_obs/test2.npy"), "Fail to set worm_obs"); // dimension mismatch
  EXPECT_ANY_THROW(wo2.ReadNpy("../gtest/model_array/worm_obs/test2.npy")); // dimension mismatch
  // EXPECT_EXIT(wo2.ReadNpy("../gtest/model_array/worm_obs/test3.npy"),
  //       ::testing::ExitedWithCode(0), "");
}

TEST(ArrWormObs, CheckReadSymm)
{
  model::ArrWormObs wo(2, 2);

  testing::internal::CaptureStdout();
  wo.ReadNpy("../gtest/model_array/worm_obs/1D_nsymm.npy");
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_STREQ("Warning!! Given array is not symmetric under the swap\nWarning!! Given array has non-zero single site operator (Cannot handle yet)\n", output.c_str()); // single site operator is not implemented yet.

  model::ArrWormObs wo2(2, 2);
  testing::internal::CaptureStdout();
  wo2.ReadNpy("../gtest/model_array/worm_obs/1D_no_onesite.npy");
  output = testing::internal::GetCapturedStdout();
  EXPECT_STREQ("", output.c_str()); // single site operator is not implemented yet.
}

TEST(ArrWormObs, CheckRead)
{
  model::ArrWormObs wo(2, 2);
  wo.ReadNpy("../gtest/model_array/worm_obs/test3.npy");
  EXPECT_EQ(wo.worm_obs() == worm_obs_check, true);
  EXPECT_FLOAT_EQ(wo(1, 1, 0, 0), 7.95);  // matrix element < 0, 0| O | 1, 1>
  EXPECT_FLOAT_EQ(wo(1, 0, 0, 1), 84.88); // matrix element < 0, 1| O | 1, 0> (left spin denote tail spin)

  testing::internal::CaptureStdout();
  wo.ReadNpy("../gtest/model_array/worm_obs/test3.npy");
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_STREQ("Worm_obs is already set\n", output.c_str());

  model::ArrWormObs wo2(4, 2);
  wo2.ReadNpy("../gtest/model_array/worm_obs/test4.npy");
  EXPECT_FLOAT_EQ(wo2(1, 0, 3, 1), 3.67); // matrix element < 3, 1 | O | 1, 0>

  model::ArrWormObs wo3(2, 2);
  wo3.ReadNpy("../gtest/model_array/worm_obs/1D_symm.npy");
  EXPECT_FLOAT_EQ(wo3(1, 0, 1, 1), 0.6307609338099869); // matrix element < 3, 1 | O | 1, 0>
}

TEST(WormObs, CheckRead)
{
  // model::WormObs wo(2, "../gtest/model_array/worm_obs/g");
  testing::internal::CaptureStdout();
  model::WormObs wo2(2, "../gtest/model_array/worm_obs/g2");
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_STREQ("Warning! : Wobs paths might be in reverse order \n", output.c_str());

  testing::internal::CaptureStdout();
  model::WormObs wo3(2, "../gtest/model_array/worm_obs/g3");
  output = testing::internal::GetCapturedStdout();
  EXPECT_STREQ("Warning!! Given array has non-zero single site operator (Cannot handle yet)\nWarning! : Only one numpy file is found. The path is set for 2 points operator \n", output.c_str());

  model::WormObs wo(2, "../gtest/model_array/worm_obs/g");
  EXPECT_EQ((*wo.first())(1, 0), 1);
  EXPECT_EQ((*wo.first())(0, 0), 0);
  EXPECT_EQ((*wo.second())(1, 0, 0, 1), 1);
  EXPECT_EQ((*wo.second())(0, 1, 1, 0), 1);

  wo.add({0, 1, 1, 0}, 10, 1, -1, -1); // spins L, sign, r, tau we don't use r and tau currently.
  wo.add({0, 1, 0, 0}, 10, 1, -1, -1);
  wo.add({0, 1, 1, 1}, 10, 1, -1, -1);

  batch_res res = wo.finalize();
  EXPECT_FLOAT_EQ((double)res.mean()[0], 11.666667); // (5 + 0 + 10) / 3

  model::WormObs wo_2site(2, "../gtest/model_array/Heisenberg/1D/original/Jz_-1_Jx_-1_Jy_-1_h_0/g");
  EXPECT_EQ((*wo_2site.first())(0, 0), 0);
  EXPECT_EQ((*wo_2site.first())(1, 0), 0);
  EXPECT_EQ((*wo_2site.second())(1, 0, 0, 1), 1);
  EXPECT_EQ((*wo_2site.second())(0, 1, 1, 0), 1);

  wo_2site.add({0, 1, 1, 0}, 10, 1, -1, -1); // spins L, sign, r, tau we don't use r and tau currently.
  wo_2site.add({0, 1, 0, 0}, 10, 1, -1, -1);
  wo_2site.add({1, 0, 0, 1}, 10, 1, -1, -1);

  batch_res res2 = wo_2site.finalize();
  EXPECT_FLOAT_EQ((double)res2.mean()[0], 10 / 3.0); // (5 + 0 + 5) / 3
}

TEST(WormObs, MapWobs)
{
  using namespace model;
  model::WormObs wo(2, "../gtest/model_array/worm_obs/g");
  model::WormObs wo_2site(2, "../gtest/model_array/Heisenberg/1D/original/Jz_-1_Jx_-1_Jy_-1_h_0/g");

  // auto a = make_pair(wo, wo);

  model::MapWormObs mwobs(std::make_pair("g", wo));
  // cout << ((*mwobs["g"].first())(0, 0)) << endl;
  EXPECT_EQ((*mwobs["g"].first())(1, 0), 1);
  model::MapWormObs mwobs2(make_pair("g1", wo), make_pair("g2", wo_2site));
  EXPECT_EQ((*mwobs2["g1"].first())(1, 0), 1);
  EXPECT_EQ((*mwobs2["g2"].second())(1, 0, 0, 1), 1);

  model::MapWormObs mwobs3(wo);
  auto it = mwobs3().begin();
  EXPECT_EQ(it->first, "G");
  EXPECT_EQ((*it->second.first())(1, 0), 1);

}

