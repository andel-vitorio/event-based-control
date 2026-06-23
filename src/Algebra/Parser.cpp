#include "../../include/Algebra/Algebra.hpp"

#include <stdexcept>
#include <cstdlib>

namespace Algebra
{

  ExpressionPtr Parser::parse(
      const std::string &expression)
  {

    Lexer lexer(expression);

    Parser parser(
        lexer.tokenize());

    return parser.parseExpression();
  }

  Parser::Parser(
      std::vector<Token> tokens)
      : tokens_(std::move(tokens))
  {
  }

  ExpressionPtr Parser::parseExpression()
  {

    auto left =
        parseTerm();

    while (
        match(TokenType::Plus) ||
        match(TokenType::Minus))
    {

      Token op =
          tokens_[position_ - 1];

      auto right =
          parseTerm();

      if (op.type == TokenType::Plus)
      {

        left =
            std::make_unique<BinaryExpression>(
                Operator::Add,
                std::move(left),
                std::move(right));
      }
      else
      {

        left =
            std::make_unique<BinaryExpression>(
                Operator::Sub,
                std::move(left),
                std::move(right));
      }
    }

    return left;
  }

  ExpressionPtr Parser::parseTerm()
  {

    auto left =
        parseFactor();

    while (
        match(TokenType::Multiply) ||
        match(TokenType::Divide))
    {

      Token op =
          tokens_[position_ - 1];

      auto right =
          parseFactor();

      if (op.type == TokenType::Multiply)
      {

        left =
            std::make_unique<BinaryExpression>(
                Operator::Mul,
                std::move(left),
                std::move(right));
      }
      else
      {

        left =
            std::make_unique<BinaryExpression>(
                Operator::Div,
                std::move(left),
                std::move(right));
      }
    }

    return left;
  }

  ExpressionPtr Parser::parseFactor()
  {

    auto left =
        parsePrimary();

    while (
        match(TokenType::Power))
    {

      auto right =
          parsePrimary();

      left =
          std::make_unique<BinaryExpression>(
              Operator::Pow,
              std::move(left),
              std::move(right));
    }

    return left;
  }

  ExpressionPtr Parser::parsePrimary()
  {

    Token token =
        peek();

    if (match(TokenType::Number))
    {

      double value =
          std::stod(token.text);

      return std::make_unique<Constant>(
          value);
    }

    if (match(TokenType::Identifier))
    {

      std::string name =
          token.text;

      if (match(TokenType::LeftParen))
      {

        std::vector<ExpressionPtr> arguments;

        arguments.push_back(
            parseExpression());

        while (match(TokenType::Comma))
        {

          arguments.push_back(
              parseExpression());
        }

        consume(
            TokenType::RightParen);

        FunctionExpression::Function function;

        if (name == "sin")
          function =
              FunctionExpression::Function::Sin;

        else if (name == "cos")
          function =
              FunctionExpression::Function::Cos;

        else if (name == "tan")
          function =
              FunctionExpression::Function::Tan;

        else if (name == "exp")
          function =
              FunctionExpression::Function::Exp;

        else if (name == "log")
          function =
              FunctionExpression::Function::Log;

        else if (name == "sqrt")
          function =
              FunctionExpression::Function::Sqrt;

        else if (name == "abs")
          function =
              FunctionExpression::Function::Abs;

        else if (name == "min")
        {
          function =
              FunctionExpression::Function::Min;
        }

        else if (name == "max")
        {
          function =
              FunctionExpression::Function::Max;
        }

        else if (name == "pow")
        {
          function =
              FunctionExpression::Function::Pow;
        }

        else
        {
          throw std::runtime_error(
              "Unknown function: " + name);
        }

        return std::make_unique<
            FunctionExpression>(
            function,
            std::move(arguments));
      }

      return std::make_unique<Variable>(
          name);
    }

    if (match(TokenType::LeftParen))
    {

      auto expression =
          parseExpression();

      consume(
          TokenType::RightParen);

      return expression;
    }

    throw std::runtime_error(
        "Invalid expression near: " + token.text);
  }

  bool Parser::match(
      TokenType type)
  {

    if (peek().type != type)
      return false;

    position_++;

    return true;
  }

  const Token &Parser::peek() const
  {

    return tokens_[position_];
  }

  Token Parser::consume(
      TokenType type)
  {

    if (peek().type != type)
    {
      throw std::runtime_error(
          "Unexpected token: " + peek().text);
    }

    return tokens_[position_++];
  }

}