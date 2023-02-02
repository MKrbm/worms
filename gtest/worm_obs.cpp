#include <autoobservable.hpp>
#include <vector>
#include <funcs.hpp>

#include "dataset.hpp"
#include "gtest/gtest.h"

using namespace std;

TEST(BaseWormObs, GetState){
  model::WormObservable wo(4, 2); //dof, legs
  EXPECT_EQ(wo.get_state({1, 2, 3, 1}), 121);
  EXPECT_EQ(wo.get_state({3, 2, 3, 1}), 123);
  EXPECT_EQ(wo.get_state({0, 1, 0, 1}), 68);
  model::WormObservable wo2(2, 2);
  EXPECT_EQ(wo2.get_state({1, 1, 0, 0}), 3);
  EXPECT_EQ(wo2.get_state({1, 0, 0, 0}), 1);
  model::WormObservable wo3(3, 2);
  EXPECT_EQ(wo3.get_state({1, 0, 1, 0}), 10);
  EXPECT_EQ(wo3.get_state({1, 1, 1, 0}), 13);
  model::WormObservable wo4(2, 1);
  EXPECT_EQ(wo4.get_state({1, 1, 0, 0}), -1);
  EXPECT_EQ(wo4.get_state({1, 0, 1, 0}), 3);
  model::WormObservable wo5(4, 1);
  EXPECT_EQ(wo5.get_state({1, 1, 3, 0}), -1);
  EXPECT_EQ(wo5.get_state({1, 0, 3, 0}), 13);
  EXPECT_EQ(wo5.get_state({1, 2, 3, 2}), 13);

}

TEST(BaseWormObs, Call){
  model::WormObservable wo(4, 1);
  EXPECT_EQ(wo({1, 2, 3, 1}), 0);
  EXPECT_EQ(wo({1, 2, 1, 1}), 0);

}

TEST(BaseWormObs, CallError){
  model::WormObservable wo(4, 1);
  // wo({1, 2, 3, 2});
  EXPECT_DEATH(double x = wo({1, 2, 1, 2}), 
  "WormObservable::operator is virtual function");
}

TEST(BaseWormObs, ErrorCase){
  model::WormObservable wo(2, 1);
  EXPECT_DEATH(wo.get_state({1, 2, 0, 0}), "spin state is out of range");
  EXPECT_DEATH(model::WormObservable(1, 3), "leg size of WormObservable must be 1 or 2");
}

TEST(NpyWormObs, GetState){
  model::NpyWormObs wo(4, 2); //dof, legs
  EXPECT_EQ(wo.get_state({1, 2, 3, 1}), 121);
  EXPECT_EQ(wo.get_state({3, 2, 3, 1}), 123);
  EXPECT_EQ(wo.get_state({0, 1, 0, 1}), 68);
  model::NpyWormObs wo2(2, 2);
  EXPECT_EQ(wo2.get_state({1, 1, 0, 0}), 3);
  EXPECT_EQ(wo2.get_state({1, 0, 0, 0}), 1);
  model::NpyWormObs wo3(3, 2);
  EXPECT_EQ(wo3.get_state({1, 0, 1, 0}), 10);
  EXPECT_EQ(wo3.get_state({1, 1, 1, 0}), 13);
  model::NpyWormObs wo4(2, 1);
  EXPECT_EQ(wo4.get_state({1, 1, 0, 0}), -1);
  EXPECT_EQ(wo4.get_state({1, 0, 1, 0}), 3);
  model::NpyWormObs wo5(4, 1);
  EXPECT_EQ(wo5.get_state({1, 1, 3, 0}), -1);
  EXPECT_EQ(wo5.get_state({1, 0, 3, 0}), 13);
  EXPECT_EQ(wo5.get_state({1, 2, 3, 2}), 13);
}

TEST(NpyWormObs, SetVector){
  model::NpyWormObs wo(4, 2); //dof, legs

  std::vector<double> v;
  for (int i=0; i < std::pow(4, 4); i++){v.push_back(i);}
  wo._SetWormObs(v);
  EXPECT_EQ(wo({1, 2, 3, 1}), 121);
}

