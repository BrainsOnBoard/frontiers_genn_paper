class LIF:public NeuronModels::Base
{
public:
  DECLARE_MODEL(LIF,5,1);
  SET_SIM_CODE(
    "const scalar alpha=($(Isyn)*$(Rmem))+$(Vrest);\n"
    "$(V)=alpha-($(ExpTC)*(alpha-$(V)));");
  SET_THRESHOLD_CONDITION_CODE("$(V) >= $(Vthresh)");
  SET_RESET_CODE("$(V)=$(Vreset);");
  SET_PARAM_NAMES({"C","TauM","Vrest","Vreset","Vthresh"});
  SET_DERIVED_PARAMS({
    {"ExpTC",[](const vector<double> &pars,double dt){return exp(-dt/pars[1]);}},
    {"Rmem",[](const vector<double> &pars,double){return pars[1]/pars[0];}}});
  SET_VARS({{"V","scalar"}});
};
IMPLEMENT_MODEL(LIF);