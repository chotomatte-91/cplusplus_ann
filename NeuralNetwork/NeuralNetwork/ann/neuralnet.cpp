#include "neuralnet.h"
#include <cassert>
#include <iostream>
#include "../utils/rng.h"

//random number generator
static RNG rng;

//default activation func does nothing
float defaultFunc(float x) { return x; }

//error function
float rmse(const std::vector<float>& predicted, const std::vector<float>& expected)
{
  assert(predicted.size() == expected.size());
  size_t N = expected.size();

  float error = 0.f;
  for (size_t i = 0; i < N; ++i)
  {
    float d = expected[i] - predicted[i];
    error += (d * d);
  }
  //return std::sqrtf(error / static_cast<float>(N));
  return (error / (2 * N));
}

Neuron::Neuron(unsigned numOutputs, unsigned index) :
  m_output(0.f),
  m_index(index),
  m_func(defaultFunc),
  m_derivative(defaultFunc)
{
  for (unsigned i = 0; i < numOutputs; ++i)
  {
    m_edges.push_back(Synapse());
    m_edges.back().weight = rng.random_prob();
    m_edges.back().deltaWeight = 0.f;
  }
}

void Neuron::setOutput(float val)
{
  m_output = val;
}

void Neuron::setWeight(unsigned to_index, float weight)
{
  m_edges[to_index].weight = weight;
}

void Neuron::setActivationFunctions(ActivationFunc f, ActivationFunc d)
{
  m_derivative = d;
  m_func = f;
}

float Neuron::getOutput() const
{
  return m_output;
}

void Neuron::calculateOutput(const Layer& previousLayer)
{
  float innerproduct = 0.f;

  //inner product of previous layer output and weights
  for (size_t n = 0; n < previousLayer.size(); ++n)
    innerproduct += previousLayer[n].getOutput() * previousLayer[n].m_edges[m_index].weight;

  //y = f(x), where x is the innerproduct calculated previously
  assert(m_func != nullptr);
  m_output = m_func(innerproduct);
}

void Neuron::computeOutputGradients(float expectedValue)
{
  float delta = expectedValue - m_output;
  m_gradient = delta * m_derivative(m_output);
}


void Neuron::computeHiddenGradients(const Layer& nextLayer)
{
  float dW = 0.f;

  //sum contribution of errors at nodes of next layer
  for (size_t i = 0; i < nextLayer.size()-1; ++i)
    dW += m_edges[i].weight * nextLayer[i].m_gradient;

  m_gradient = dW * m_derivative(m_output);
}

void Neuron::updateWeights(Layer& prevLayer, float alpha)
{
  //update the weights in the previous layer
  for (size_t i = 0; i < prevLayer.size(); ++i)
  {
    Neuron& prevNeuron = prevLayer[i];
    float new_deltaWeight = alpha * m_gradient * prevNeuron.getOutput();

    prevNeuron.m_edges[m_index].deltaWeight = new_deltaWeight;
    prevNeuron.m_edges[m_index].weight += new_deltaWeight;
  }
}

Neuron& NeuralNet::getNeuron(unsigned layerIndex, unsigned index)
{
  assert(m_layers.size() != 0);
  return m_layers[layerIndex][index];
}

NeuralNet::NeuralNet(const std::vector<unsigned>& config)
{
  size_t layerCount = config.size();
  //create layers
  for (size_t l = 0; l < layerCount; ++l)
  {
    //output layer have no outputs
    unsigned numOuts = (l == layerCount-1) ? 0 : config[l+1];

    m_layers.push_back(Layer());

    //extra layer for bias neuron
    for (size_t n = 0; n <= config[l]; ++n)
      m_layers.back().emplace_back(numOuts, n);

    //set output of bias neuron to 1.0, nothing will change its output value
    m_layers.back().back().setOutput(1.f);
  }
}

float NeuralNet::forward(const std::vector<float>& inputs)
{
  assert(inputs.size() == m_layers[0].size()-1);

  //set inputs as output values of initial layer
  Layer& inputLayer = m_layers[0];
  for (size_t i = 0; i < inputs.size(); ++i)
    inputLayer[i].setOutput(inputs[i]);

  //feed forward, starting from first hidden layer
  for (size_t l = 1; l < m_layers.size(); ++l)
  {
    Layer& currentLayer = m_layers[l];
    Layer& previousLayer = m_layers[l-1];
    
    //update weights for each neuron, ignoring the bias neuron 
    for (size_t n = 0; n < currentLayer.size()-1; ++n)
      currentLayer[n].calculateOutput(previousLayer);
  }

  //get output (assume only one neuron in output layer) (extra bias neuron ignored)
  assert(m_layers.back().size() == 2);

  float result = m_layers.back()[0].getOutput();

  std::cout << result << std::endl;

  return result;
}


float NeuralNet::back(const std::vector<float>& predicted, const std::vector<float>& labels, float alpha)
{
  //calculate rmse
  float err = rmse(predicted, labels);

  //calculate gradient at output layer
  Layer& outputLayer = m_layers.back();
  for (size_t i = 0; i < outputLayer.size()-1; ++i)
    outputLayer[i].computeOutputGradients(labels[i]);

  //calculate gradients of hidden layers
  for (size_t i = m_layers.size() - 2; i > 0; --i)
  {
    Layer& hiddenLayer = m_layers[i];
    Layer& nextLayer = m_layers[i+1];

    for (size_t n = 0; n < hiddenLayer.size(); ++n)
      hiddenLayer[n].computeHiddenGradients(nextLayer);

  }

  //update weights of layers using gradients
  for (size_t i = m_layers.size() - 1; i > 0; --i)
  {
    Layer& currentLayer = m_layers[i];
    Layer& prevLayer= m_layers[i-1];

    for(size_t n = 0; n < currentLayer.size()-1; ++n)
      currentLayer[n].updateWeights(prevLayer, alpha);
  }
  
  return err;
}

void NeuralNet::train(const Matrix& inputs, const std::vector<float>& labels, float learningRate, unsigned numIterations)
{
  //number of rows = number of inputs
  size_t numInputs = inputs.size();

  //every input should have a label
  assert(numInputs == labels.size());

  for (unsigned i = 0; i < numIterations; ++i)
  {
    std::vector<float> outputs(labels.size(), 0.f);

    //feed forward
    for (size_t j = 0; j < numInputs; ++j)
      outputs[j] = forward(inputs[j]);

    //back propagate
    float error = back(outputs, labels, learningRate);

    //output error
    std::cout << "Error in " << i+1 << " iteration: " << error << std::endl; 
  }
}

float NeuralNet::predict(const std::vector<float>& input)
{
  return forward(input);
}

size_t NeuralNet::numLayers() const
{
  return m_layers.size();
}

size_t NeuralNet::numNeurons(unsigned index) const
{
  return m_layers[index].size();
}