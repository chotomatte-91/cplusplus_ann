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

//currently this network does not allow user to specify input-output configuration,
//assumes output layer has only one neuron 
class Neuron
{
public:
  Neuron(unsigned numOutputs, unsigned index);
  void calculateOutput(const Layer& previousLayer);
  void computeOutputGradients(float expectedValue);
  void computeHiddenGradients(const Layer& nextLayer);
  void updateWeights(Layer& previousLayer, float a);
  
  void setWeight(unsigned to_index, float weight);
  void setOutput(float val);
  void setActivationFunctions(ActivationFunc func, ActivationFunc derivative);
  float getOutput() const;

private:
  ActivationFunc m_func;
  ActivationFunc m_derivative;
  float m_output;
  float m_gradient;
  unsigned m_index;
  std::vector<Synapse> m_edges;
};


using Matrix = std::vector<std::vector<float>>;
class NeuralNet
{
public:
  NeuralNet(const std::vector<unsigned>& config);
  Neuron& getNeuron(unsigned layerIndex, unsigned index);
  void train(const Matrix& inputs, const std::vector<float>& labels, float learningRate, unsigned numIter);
  float predict(const std::vector<float>& input);
  size_t numLayers() const;
  size_t numNeurons(unsigned layerIndex) const;

private:
  float forward(const std::vector<float>& inputvalues);
  float back(const std::vector<float>& predicted, const std::vector<float>& labels, float alpha);

  std::vector<Layer> m_layers;
};

#endif
