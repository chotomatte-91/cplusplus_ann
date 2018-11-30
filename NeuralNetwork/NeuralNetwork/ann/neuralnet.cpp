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

void Neuron::updateWeights(const Layer& previousLayer)
{
  float innerproduct = 0.f;

  //inner product of previous layer output and weights
  for (size_t n = 0; n < previousLayer.size(); ++n)
    innerproduct += previousLayer[n].getOutput() * previousLayer[n].m_edges[m_index].weight;

  //y = f(x), where x is the innerproduct calculated previously
  assert(m_func != nullptr);
  m_output = m_func(innerproduct);
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
    {
      m_layers.back().emplace_back(numOuts, n);
    }
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
    
    //update weights for each neuron
    for (size_t n = 0; n < currentLayer.size(); ++n)
      currentLayer[n].updateWeights(previousLayer);
  }


}