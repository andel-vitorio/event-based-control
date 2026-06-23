#include "../../include/Algebra/Algebra.hpp"

#include <cmath>
#include <stdexcept>

namespace Algebra
{

  Constant::Constant(double value)
      : value_(value)
  {
  }

  double Constant::evaluate(
      const Variables &,
      double) const
  {
    return value_;
  }

  std::string Constant::str() const
  {
    return std::to_string(value_);
  }

  Variable::Variable(
      std::string name)
      : name_(std::move(name))
  {
  }

  double Variable::evaluate(
      const Variables &variables,
      double t) const
  {

    if (name_ == "t")
      return t;

    auto it =
        variables.find(name_);

    if (it == variables.end())
    {
      throw std::runtime_error(
          "Unknown variable: " + name_);
    }

    return it->second;
  }

  std::string Variable::str() const
  {
    return name_;
  }

  BinaryExpression::BinaryExpression(
      Operator op,
      ExpressionPtr left,
      ExpressionPtr right)
      : op_(op),
        left_(std::move(left)),
        right_(std::move(right))
  {
  }

  double BinaryExpression::evaluate(
      const Variables &variables,
      double t) const
  {

    double left =
        left_->evaluate(
            variables, t);

    double right =
        right_->evaluate(
            variables, t);

    switch (op_)
    {

    case Operator::Add:
      return left + right;

    case Operator::Sub:
      return left - right;

    case Operator::Mul:
      return left * right;

    case Operator::Div:
      return left / right;

    case Operator::Pow:
      return std::pow(left, right);
    }

    return 0;
  }

  std::string BinaryExpression::str() const
  {
    return "(" +
           left_->str() +
           " op " +
           right_->str() +
           ")";
  }

  FunctionExpression::FunctionExpression(
      Function function,
      std::vector<ExpressionPtr> arguments)
      : function_(function),
        arguments_(std::move(arguments))
  {
  }

  double FunctionExpression::evaluate(
      const Variables &variables,
      double t) const
  {

    std::vector<double> values;

    for (auto &arg : arguments_)
    {
      values.push_back(
          arg->evaluate(
              variables,
              t));
    }

    switch (function_)
    {

    case Function::Sin:
      return std::sin(values[0]);

    case Function::Cos:
      return std::cos(values[0]);

    case Function::Tan:
      return std::tan(values[0]);

    case Function::Exp:
      return std::exp(values[0]);

    case Function::Log:
      return std::log(values[0]);

    case Function::Sqrt:
      return std::sqrt(values[0]);

    case Function::Abs:
      return std::abs(values[0]);

    case Function::Min:
      return std::min(
          values[0],
          values[1]);

    case Function::Max:
      return std::max(
          values[0],
          values[1]);

    case Function::Pow:
      return std::pow(
          values[0],
          values[1]);
    }

    return 0;
  }

  std::string FunctionExpression::str() const
  {

    std::string result =
        name_ + "(";

    for (size_t i = 0; i < arguments_.size(); ++i)
    {

      result += arguments_[i]->str();

      if (i + 1 < arguments_.size())
        result += ", ";
    }

    result += ")";

    return result;
  }

  ConstantSymbol::ConstantSymbol(
      Type type)
      : type_(type)
  {
  }

  double ConstantSymbol::evaluate(
      const Variables &,
      double) const
  {

    switch (type_)
    {
    case Type::Pi:
      return std::acos(-1.0);
    case Type::E:
      return std::exp(1.0);
    }

    return 0.0;
  }

  std::string ConstantSymbol::str() const
  {

    switch (type_)
    {

    case Type::Pi:
      return "pi";

    case Type::E:
      return "e";
    }

    return "";
  }
}