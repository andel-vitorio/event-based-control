#include "../../include/Algebra/Algebra.hpp"

#include <cctype>
#include <stdexcept>

namespace Algebra
{

  Lexer::Lexer(
      const std::string &expression)
      : text_(expression)
  {
  }

  char Lexer::current() const
  {
    if (position_ >= text_.size())
      return '\0';

    return text_[position_];
  }

  void Lexer::advance()
  {
    position_++;
  }

  void Lexer::skipSpaces()
  {
    while (std::isspace(current()))
      advance();
  }

  Token Lexer::number()
  {

    std::string value;

    while (
        std::isdigit(current()) ||
        current() == '.')
    {

      value += current();

      advance();
    }

    return {
        TokenType::Number,
        value};
  }

  Token Lexer::identifier()
  {

    std::string name;

    while (
        std::isalnum(current()) ||
        current() == '_')
    {

      name += current();

      advance();
    }

    return {
        TokenType::Identifier,
        name};
  }

  std::vector<Token> Lexer::tokenize()
  {

    std::vector<Token> tokens;

    while (current() != '\0')
    {

      skipSpaces();

      char c = current();

      if (std::isdigit(c))
      {
        tokens.push_back(number());
        continue;
      }

      if (std::isalpha(c))
      {
        tokens.push_back(identifier());
        continue;
      }

      advance();

      switch (c)
      {

      case '+':
        tokens.push_back({TokenType::Plus, "+"});
        break;

      case '-':
        tokens.push_back({TokenType::Minus, "-"});
        break;

      case '*':
        tokens.push_back({TokenType::Multiply, "*"});
        break;

      case '/':
        tokens.push_back({TokenType::Divide, "/"});
        break;

      case '^':
        tokens.push_back({TokenType::Power, "^"});
        break;

      case '(':
        tokens.push_back({TokenType::LeftParen, "("});
        break;

      case ')':
        tokens.push_back({TokenType::RightParen, ")"});
        break;

      case ',':
        tokens.push_back({TokenType::Comma, ","});
        break;

      default:

        throw std::runtime_error(
            "Invalid character: " + std::string(1, c));
      }
    }

    tokens.push_back(
        {TokenType::End, ""});

    return tokens;
  }

}