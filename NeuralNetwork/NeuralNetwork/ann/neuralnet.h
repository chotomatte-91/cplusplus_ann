#ifndef _NEURAL_NET_H
#define _NEURAL_NET_H
#pragma once

#include <vector>

class Neuron;
struct Synapse
{
  float weight;
  unsigned from;
  unsigned to;
};

using ActivationFunc = float(*)(float);
using Layer = std::vector<Neuron>;

class Neuron
{
public:
  Neuron(unsigned numOutputs, unsigned index);
  void updateWeights(const Layer& previousLayer);
  void setOutput(float val);
  void setActivationFunctions(ActivationFunc func, ActivationFunc derivative);
  float getOutput() const;

private:
  ActivationFunc m_func = nullptr;
  ActivationFunc m_derivative = nullptr;
  float m_output;
  unsigned m_index;
  std::vector<Synapse> m_edges;
};



class NeuralNet
{
public:
  NeuralNet(const std::vector<unsigned>& config);
  Neuron& getNeuron(unsigned layerNum, unsigned index);
  std::vector<float> predict(const std::vector<float>& inputs, const std::vector<float>& labels);
  

private:
  void forward(const std::vector<float>& inputvalues);
  void back(const std::vector<float>& labels);
  std::vector<float> output() const;

  std::vector<Layer> m_layers;
};


#endif
