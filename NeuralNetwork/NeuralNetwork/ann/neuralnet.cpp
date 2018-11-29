#include "neuralnet.h"
#include "../utils/rng.h"

//random number generator
static RNG rng = RNG();

Neuron::Neuron(unsigned numOutputs, const float& val) :
  m_value(val)
{
  for (unsigned i = 0; i < numOutputs; ++i)
  {
    m_edges.push_back(Synapse());
    m_edges.back().weight = rng.random_prob();
  }
}

void Layer::AddNeuron(unsigned numOut, float v)
{
  float new_val = 0.f;
  if (v == 0.f)
  {
    //random value
  }
  else
    new_val = v;

  m_neurons.emplace_back(numOut, new_val);
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
      m_layers.back().AddNeuron();
    }
  }
}