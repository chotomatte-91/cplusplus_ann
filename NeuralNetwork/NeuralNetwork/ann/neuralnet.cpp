#include "neuralnet.h"
#include <cassert>
#include "../utils/rng.h"

//random number generator
static RNG rng;

//random number generator
Neuron::Neuron(unsigned numOutputs, unsigned index) :
  m_output(0.f),
  m_index(index)
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
  for (size_t i = 0; i < nextLayer.size(); ++i)
    dW += m_edges[i].weight * nextLayer[i].m_gradient;

  m_gradient = dW * m_derivative(m_output);
}

void Neuron::updateWeights(Layer& prevLayer)
{
  float alpha = 0.f;

  //update the weights in the previous layer
  for (size_t i = 0; i < prevLayer.size(); ++i)
  {
    Neuron& prevNeuron = prevLayer[i];
    float new_deltaWeight = alpha * m_gradient * prevNeuron.getOutput();

    prevNeuron.m_edges[m_index].deltaWeight = new_deltaWeight;
    prevNeuron.m_edges[m_index].weight += new_deltaWeight;
  }
}

Neuron& NeuralNet::getNeuron(unsigned layerNum, unsigned index)
{
  assert(m_layers.size() != 0);
  return m_layers[layerNum][index];
}

NeuralNet::NeuralNet(const std::vector<unsigned>& config)
{
  size_t layerCount = config.size();
  //create layers
  for (size_t l = 0; l < layerCount; ++l)
  {
    //output layer have no outputs
    unsigned numOuts = l == layerCount-1 ? 0 : config[l+1];

    m_layers.push_back(Layer());

    //extra layer for bias neuron
    for (size_t n = 0; n <= config[l]; ++n)
      m_layers.back().emplace_back(numOuts, n);

    //set output of bias neuron to 1.0, nothing will change its output value
    m_layers.back().back().setOutput(1.f);
  }
}

void NeuralNet::forward(const std::vector<float>& inputs)
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
}

float NeuralNet::rmse(const std::vector<float>& expectedValues) const
{
  const Layer& outputLayer = m_layers.back();
  float error = 0.f;

  assert(expectedValues.size() == outputLayer.size()-1);
  
  for (size_t i = 0; i < expectedValues.size(); ++i)
  {
    float d  = expectedValues[i] - outputLayer[i].getOutput();
    error += (d * d);
  }
  return std::sqrtf(error / float(expectedValues.size()));
}

void NeuralNet::back(const std::vector<float>& labels)
{
  //calculate rmse
  float err = rmse(labels);

  //calculate gradient at output layer
  Layer& outputLayer = m_layers.back();
  for (size_t i = 0; i < outputLayer.size(); ++i)
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
      currentLayer[n].updateWeights(prevLayer);
  }

}

std::vector<float> NeuralNet::output() const
{
  //copy values from output layer and return
  const Layer& outputLayer = m_layers.back();
  std::vector<float> result;
  result.reserve(outputLayer.size()-1);

  for (size_t i = 0; i < outputLayer.size() - 1; ++i)
    result.push_back(outputLayer[i].getOutput());

  return result;
}

std::vector<float> NeuralNet::predict(const std::vector<float>& inputs, const std::vector<float>& labels, unsigned numIterations)
{
  for (unsigned i = 0; i < numIterations; ++i)
  {
    forward(inputs);
    back(labels);
  }
  return output();
}