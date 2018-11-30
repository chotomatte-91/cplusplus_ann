#ifndef _NEURAL_NET_H
#define _NEURAL_NET_H
#pragma once

#include <vector>

struct Synapse
{
  float weight;
  unsigned from;
  unsigned to;
};

class Neuron
{
public:
  Neuron(unsigned numOutputs, const float& val);
private:
  float m_value;
  std::vector<Synapse> m_edges;
};

class Layer
{
public:
  void AddNeuron(unsigned numOutputs, float value=0.f);

private:
  std::vector<Neuron> m_neurons;
};

class NeuralNet
{
public:
  NeuralNet(const std::vector<unsigned>& config);
  void forward(const std::vector<float>& inputvalues);
  void back(const std::vector<float>& labels);
  std::vector<float> predict() const;


private:
  std::vector<Layer> m_layers;

};

#endif
