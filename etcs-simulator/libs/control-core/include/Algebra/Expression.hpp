#pragma once

#include <memory>
#include <string>
#include <map>
#include <functional>
#include <vector>

namespace Algebra
{

  using Variables =
      std::map<std::string, double>;

  class Expression
  {

  public:
    virtual ~Expression() = default;

    virtual double evaluate(
        const Variables &variables,
        double t) const = 0;

    virtual std::string str() const = 0;
  };

  using ExpressionPtr =
      std::unique_ptr<Expression>;

  class Constant : public Expression
  {

  public:
    explicit Constant(double value);

    double evaluate(
        const Variables &,
        double) const override;

    std::string str() const override;

  private:
    double value_;
  };

  class ConstantSymbol : public Expression
  {

  public:
    enum class Type
    {
      Pi,
      E
    };

    explicit ConstantSymbol(
        Type type);

    double evaluate(
        const Variables &,
        double) const override;

    std::string str() const override;

  private:
    Type type_;
  };

  class Variable : public Expression
  {

  public:
    explicit Variable(
        std::string name);

    double evaluate(
        const Variables &variables,
        double t) const override;

    std::string str() const override;

  private:
    std::string name_;
  };

  enum class Operator
  {
    Add,
    Sub,
    Mul,
    Div,
    Pow
  };

  class BinaryExpression : public Expression
  {

  public:
    BinaryExpression(
        Operator op,
        ExpressionPtr left,
        ExpressionPtr right);

    double evaluate(
        const Variables &variables,
        double t) const override;

    std::string str() const override;

  private:
    Operator op_;

    ExpressionPtr left_;

    ExpressionPtr right_;
  };

  class FunctionExpression : public Expression
  {

  public:
    enum class Function
    {
      Sin,
      Cos,
      Tan,

      Exp,
      Log,
      Sqrt,

      Abs,

      Min,
      Max,
      Pow
    };

    FunctionExpression(
        Function function,
        std::vector<ExpressionPtr> arguments);

    double evaluate(
        const Variables &variables,
        double t) const override;

    std::string str() const override;

  private:
    Function function_;
    std::string name_;
    std::vector<ExpressionPtr> arguments_;
  };
}