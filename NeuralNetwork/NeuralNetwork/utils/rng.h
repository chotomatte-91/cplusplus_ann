#ifndef _RNG_H
#define _RNG_H
#pragma once

#include <limits>
#include <random>

class RNG
{
public:
  RNG() :
    m_engine(m_rd()),
    m_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max()),
    m_prob_dist(0.f, 1.f)
  {}

  float rand()
  {
    return m_dist(m_engine);
  }

  float random_prob()
  {
    return m_prob_dist(m_engine);
  }

private:
  std::random_device m_rd;
  std::mt19937 m_engine;
  std::uniform_real_distribution<float> m_dist;
  std::uniform_real_distribution<float> m_prob_dist;
};

#endif

