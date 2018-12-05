#include "neuralnet.h"
#include <cassert>
#include <iostream>
#include <string>
#include "../utils/rng.h"


//random number generator
static RNG rng;

//default activation func does nothing
float identity(float x) { return x; }
float identityDerivative(float x) { return 1.f; }

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
  m_input(0.f),
  m_index(index),
  m_func(identity),
  m_derivative(identityDerivative)
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
    innerproduct += (previousLayer[n].getOutput() * previousLayer[n].m_edges[m_index].weight);

  //y = f(x), where x is the innerproduct calculated previously
  m_input = innerproduct;
  assert(m_func != nullptr);
  m_output = m_func(m_input);
}

void Neuron::computeOutputGradients(float expectedValue)
{
  float delta = expectedValue - m_output;
  m_gradient = delta * m_derivative(m_input);
}


void Neuron::computeHiddenGradients(const Layer& nextLayer)
{
  float dW = 0.f;

  //sum contribution of errors at nodes of next layer
  for (size_t i = 0; i < nextLayer.size()-1; ++i)
    dW += m_edges[i].weight * nextLayer[i].m_gradient;

  m_gradient = dW * m_derivative(m_input);
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

void Neuron::status() const
{
  std::string toPrint = "Neuron " + std::to_string(m_index);
  std::cout << "------------------------------" << toPrint << "-----------------------------" << std::endl;
  std::cout << "Output: " << m_output << std::endl;
  std::cout << "Gradient: " << m_gradient << std::endl;
}

std::vector<float> Neuron::getWeights() const
{
  std::vector<float> weights(m_edges.size(), 0.f);
  for (size_t i = 0; i < m_edges.size(); ++i)
    weights[i] = m_edges[i].weight;
  return weights;
}

float Neuron::getWeight(unsigned toNeuronIndex) const
{
  return m_edges[toNeuronIndex].weight;
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

void NeuralNet::status()
{
  size_t N = numLayers();

  for (size_t i = 0; i < N; ++i)
  {
    if (i == 0)
      std::cout << "INPUT LAYER" << std::endl;
    else if (i == N - 1)
      std::cout << "OUTPUT LAYER" << std::endl;
    else
      std::cout << "HIDDEN LAYER" << std::endl;

    for (size_t j = 0; j < numNeurons(i); ++j)
      getNeuron(i, j).status();

    std::cout << "-------------------------------------------------------------------" << std::endl;
  }
}

void NeuralNet::printWeights()
{
  float i1h1 = getNeuron(0, 0).getWeight(0);
  float i2h1 = getNeuron(0, 1).getWeight(0);
  float h1_bias = getNeuron(0, 2).getWeight(0);

  float i1h2 = getNeuron(0, 0).getWeight(1);
  float i2h2 = getNeuron(0, 1).getWeight(1);
  float h2_bias = getNeuron(0, 2).getWeight(1);

  float h1o1 = getNeuron(1, 0).getWeight(0);
  float h2o1 = getNeuron(1, 1).getWeight(0);
  float o1_bias = getNeuron(1, 2).getWeight(0);

  std::cout << "Input layer to hidden layer weights" << std::endl;
  std::cout << "[" << h1_bias << ", " << i1h1 << ", " << i2h1 << "] ";
  std::cout << "[" << h2_bias << ", " << i1h2 << ", " << i2h2 << "]" << std::endl;
  std::cout << "hidden layer to output layer weights" << std::endl;
  std::cout << "[" << o1_bias << ", " << h1o1 << ", " << h2o1 << "]" << std::endl;
}