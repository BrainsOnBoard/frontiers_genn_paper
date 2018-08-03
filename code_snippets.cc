// LIF Neuron model
class LIF:public NeuronModels::Base
{
public:
  DECLARE_MODEL(LIF,1,1);
  SET_SIM_CODE("$(V)=($(Isyn)*$(TauM)*(1.0-$(ExpTC)))+($(ExpTC)*$(V));\n");
  SET_THRESHOLD_CONDITION_CODE("$(V)>=1.0");
  SET_RESET_CODE("$(V)=0.0;");
  SET_PARAM_NAMES({"TauM"});
  SET_DERIVED_PARAMS({
      {"ExpTC",[](const vector<double> &pars,double dt)
               {return exp(-dt/pars[0]);}}});
  SET_VARS({{"V","scalar"}});
};
IMPLEMENT_MODEL(LIF);
// Neuron population
InitVarSnippet::Uniform::ParamValues vDist(0.0,1.0);
LIF::ParamValues params(20.0);
LIF::VarValues initState(initVar<InitVarSnippet::Uniform>(vDist));
model.addNeuronPopulation<LIF>("pop",1000,params,initState);