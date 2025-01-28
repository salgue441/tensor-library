#include <gtest/gtest.h>
#include <tf/math/random.hpp>
#include <tf/math/utils.hpp>
#include <vector>
#include <thread>
#include <numeric>

using namespace tf::math;

namespace test
{
  class MathTest : public ::testing::Test
  {
  protected:
    void SetUp() override
    {
      RandomGenerator::instance().set_seed(42);
    }
  };

  TEST_F(MathTest, RandomUniform)
  {
    float val = RandomGenerator::instance().uniform<float>(
        0.0f, 1.0f);

    EXPECT_GE(val, 0.0f);
    EXPECT_LE(val, 1.0f);

    std::vector<float> data(1000);
    RandomGenerator::instance().fill_uniform(data.data(), data.size(),
                                             -1.0f, 1.0f);

    for (float val : data)
    {
      EXPECT_GE(val, -1.0f);
      EXPECT_LE(val, 1.0f);
    }
  }

  TEST_F(MathTest, RandomNormal)
  {
    float val = RandomGenerator::instance().normal<float>(
        0.0f, 1.0f);

    EXPECT_NE(val, 0.0f);

    std::vector<float> data(1000);
    RandomGenerator::instance().fill_normal(data.data(), data.size(),
                                            0.0f, 1.0f);

    float mean = std::accumulate(data.begin(), data.end(),
                                 0.0f) /
                 data.size();

    EXPECT_NEAR(mean, 0.0f, 0.1f);
  }

  TEST_F(MathTest, RandomBernoulli)
  {
    const int trials = 1000;
    int count = 0;

    for (int i = 0; i < trials; ++i)
    {
      if (RandomGenerator::instance().bernoulli(0.5))
      {
        count++;
      }
    }

    EXPECT_NEAR(static_cast<float>(count) / trials, 0.5f, 0.1f);
  }

  TEST_F(MathTest, BasicMathOperations)
  {
    EXPECT_EQ(clamp(5.0f, 0.0f, 10.0f), 5.0f);
    EXPECT_EQ(clamp(-1.0f, 0.0f, 10.0f), 0.0f);
    EXPECT_EQ(clamp(11.0f, 0.0f, 10.0f), 10.0f);

    EXPECT_FLOAT_EQ(lerp(0.0f, 10.0f, 0.5f), 5.0f);
    EXPECT_FLOAT_EQ(lerp(0.0f, 1.0f, 0.25f), 0.25f);
  }

  TEST_F(MathTest, ActivationFunctions)
  {
    EXPECT_FLOAT_EQ(sigmoid(0.0f), 0.5f);
    EXPECT_FLOAT_EQ(sigmoid(-1.0f), 1.0f / (1.0f + std::exp(1.0f)));
    EXPECT_FLOAT_EQ(sigmoid(1.0f), 1.0f / (1.0f + std::exp(-1.0f)));

    EXPECT_FLOAT_EQ(tanh(0.0f), 0.0f);
    EXPECT_FLOAT_EQ(tanh(-1.0f), std::tanh(-1.0f));
    EXPECT_FLOAT_EQ(tanh(1.0f), std::tanh(1.0f));

    EXPECT_FLOAT_EQ(relu(0.0f), 0.0f);
    EXPECT_FLOAT_EQ(relu(-1.0f), 0.0f);
    EXPECT_FLOAT_EQ(relu(1.0f), 1.0f);

    EXPECT_FLOAT_EQ(leaky_relu(0.0f), 0.0f);
    EXPECT_FLOAT_EQ(leaky_relu(-1.0f), -0.01f);
    EXPECT_FLOAT_EQ(leaky_relu(1.0f), 1.0f);
  }

  TEST_F(MathTest, ActivationFunctionDerivatives)
  {
    EXPECT_FLOAT_EQ(sigmoid_derivative(0.0f), 0.25f);
    EXPECT_FLOAT_EQ(sigmoid_derivative(-1.0f), std::exp(1.0f) / std::pow(1.0f + std::exp(1.0f), 2));
    EXPECT_FLOAT_EQ(sigmoid_derivative(1.0f), std::exp(-1.0f) / std::pow(1.0f + std::exp(-1.0f), 2));

    EXPECT_FLOAT_EQ(tanh_derivative(0.0f), 1.0f);
    EXPECT_FLOAT_EQ(tanh_derivative(-1.0f), 1.0f - std::pow(std::tanh(-1.0f), 2));
    EXPECT_FLOAT_EQ(tanh_derivative(1.0f), 1.0f - std::pow(std::tanh(1.0f), 2));

    EXPECT_FLOAT_EQ(relu_derivative(0.0f), 0.0f);
    EXPECT_FLOAT_EQ(relu_derivative(-1.0f), 0.0f);
    EXPECT_FLOAT_EQ(relu_derivative(1.0f), 1.0f);

    EXPECT_FLOAT_EQ(leaky_relu_derivative(0.0f), 0.0f);
    EXPECT_FLOAT_EQ(leaky_relu_derivative(-1.0f), 0.01f);
    EXPECT_FLOAT_EQ(leaky_relu_derivative(1.0f), 1.0f);
  }

  TEST_F(MathTest, MeanVarianceStddev)
  {
    std::vector<float> data(1000);
    RandomGenerator::instance().fill_uniform(data.data(), data.size(),
                                             0.0f, 1.0f);

    float m = mean(data.begin(), data.end());
    float v = variance(data.begin(), data.end());
    float s = stddev(data.begin(), data.end());

    EXPECT_NEAR(m, 0.5f, 0.1f);
    EXPECT_NEAR(v, 1.0f / 12.0f, 0.1f);
    EXPECT_NEAR(s, std::sqrt(1.0f / 12.0f), 0.1f);
  }

  TEST_F(MathTest, Covariance)
  {
    std::vector<float> data1(1000);
    std::vector<float> data2(1000);
    RandomGenerator::instance().fill_uniform(data1.data(), data1.size(),
                                             0.0f, 1.0f);
    RandomGenerator::instance().fill_uniform(data2.data(), data2.size(),
                                             0.0f, 1.0f);

    float cov = covariance(data1.begin(), data1.end(),
                           data2.begin(), data2.end());

    EXPECT_NEAR(cov, 0.0f, 0.1f);

    std::vector<float> data3(1000);

    RandomGenerator::instance().fill_uniform(data3.data(), data3.size(),
                                             0.0f, 1.0f);

    float cov2 = covariance(data1.begin(), data1.end(),
                            data3.begin(), data3.end());

    EXPECT_NEAR(cov2, 0.0f, 0.1f);
  }

  TEST_F(MathTest, Correlation)
  {
    std::vector<float> data1(1000);
    std::vector<float> data2(1000);
    RandomGenerator::instance().fill_uniform(data1.data(), data1.size(),
                                             0.0f, 1.0f);
    RandomGenerator::instance().fill_uniform(data2.data(), data2.size(),
                                             0.0f, 1.0f);

    float corr = correlation(data1.begin(), data1.end(),
                             data2.begin(), data2.end());

    EXPECT_NEAR(corr, 0.0f, 0.1f);

    std::vector<float> data3(1000);

    RandomGenerator::instance().fill_uniform(data3.data(), data3.size(),
                                             0.0f, 1.0f);

    float corr2 = correlation(data1.begin(), data1.end(),
                              data3.begin(), data3.end());

    EXPECT_NEAR(corr2, 0.0f, 0.1f);
  }

  TEST_F(MathTest, ThreadSafety)
  {
    std::vector<float> data(1000);
    std::vector<std::thread> threads;

    for (int i = 0; i < 10; ++i)
    {
      threads.emplace_back([&data]()
                           { RandomGenerator::instance().fill_uniform(data.data(), data.size(),
                                                                      0.0f, 1.0f); });
    }

    for (auto &thread : threads)
    {
      thread.join();
    }
  }

} // namespace test