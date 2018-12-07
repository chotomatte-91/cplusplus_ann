#ifndef _NEURAL_NET_H
#define _NEURAL_NET_H
#pragma once

#include <vector>

template <typename T>
class NeuralNet
{
public:
  struct Synapse
  {
    T weight;
    //T deltaWeight;
  };

  class Neuron;
  using ActivationFunc = T(*)(T);
  using ErrorFunc = T (*)(const std::vector<T>&, const std::vector<T>&);
  using Layer = std::vector<Neuron>;
  using Matrix = std::vector<std::vector<T>>;

  //currently this network does not allow user to specify input-output configuration,
  //assumes output layer has only one neuron 
  class Neuron
  {
  public:
    Neuron(unsigned numOutputs, unsigned index);
    void calculateOutput(const Layer& previousLayer);
    void computeOutputGradients(const T& errorDelta);
    void computeHiddenGradients(const Layer& nextLayer);
    void updateWeights(Layer& previousLayer, float a);
    void setWeight(unsigned nextLayerNeuronIndex, const T& weight);
    void setOutput(const T& val);
    void setActivationFunctions(ActivationFunc func, ActivationFunc derivative);
    void status() const;

	  size_t getNumSynapse() const;
    unsigned getIndex()const;
	  T getSynapseWeight(unsigned index) const;
    T getWeight(unsigned toNeuronIndex) const;
    T useFunc(const T& val) const;
    T useDerivative(const T& val) const;
    T getOutput() const;
    std::vector<T> getWeights() const;

  private:
    ActivationFunc m_func;
    ActivationFunc m_derivative;
    T m_output;
    T m_input;
    T m_gradient;
    unsigned m_index;
    std::vector<Synapse> m_edges;
  };

  NeuralNet(const std::vector<unsigned>& config);
  Neuron& getNeuron(unsigned layerIndex, unsigned index);
  void train(const Matrix& inputs, const std::vector<T>& labels, float learningRate, unsigned numIter);
  void Forward(unsigned current);
  void SetOutputAtLayer(unsigned index, const std::vector<float>& inputs);
  void GetOutputAtLayer(unsigned index, std::vector<float>& outputs);
  Layer& GetLayer(unsigned index);

  T predict(const std::vector<T>& input);
  size_t numLayers() const;
  size_t numNeurons(unsigned layerIndex) const;
  void status();
  void printWeights();
  void setErrorFunctions(ErrorFunc error_function, ErrorFunc error_func_derivative);

private:
  T forward(const std::vector<T>& inputvalues);
  T back(const std::vector<T>& predicted, const std::vector<T>& labels, float alpha);

  ErrorFunc m_ef;
  ErrorFunc m_ef_prime;
  std::vector<Layer> m_layers;
};

#include "neuralnet.cpp"
#endif
