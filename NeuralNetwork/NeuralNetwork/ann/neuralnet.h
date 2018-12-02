#ifndef _NEURAL_NET_H
#define _NEURAL_NET_H
#pragma once

#include <vector>

class Neuron;
struct Synapse
{
  float weight;
  float deltaWeight;
};

using ActivationFunc = float(*)(float);
using Layer = std::vector<Neuron>;

class Neuron
{
public:
  Neuron(unsigned numOutputs, unsigned index);
  void calculateOutput(const Layer& previousLayer);
  void computeOutputGradients(float expectedValue);
  void computeHiddenGradients(const Layer& nextLayer);
  void updateWeights(Layer& previousLayer);
  
  void setOutput(float val);
  void setActivationFunctions(ActivationFunc func, ActivationFunc derivative);
  float getOutput() const;

private:
  ActivationFunc m_func = nullptr;
  ActivationFunc m_derivative = nullptr;
  float m_output;
  float m_gradient;
  unsigned m_index;
  std::vector<Synapse> m_edges;
};



class NeuralNet
{
public:
  NeuralNet(const std::vector<unsigned>& config);
  Neuron& getNeuron(unsigned layerNum, unsigned index);
  std::vector<float> predict(const std::vector<float>& inputs, const std::vector<float>& labels, unsigned numIter);

private:
  void forward(const std::vector<float>& inputvalues);
  void back(const std::vector<float>& labels);
  float rmse(const std::vector<float>& expectedValues) const;
  std::vector<float> output() const;

  std::vector<Layer> m_layers;
};

#endif
