#pragma once

#include "Expression.hpp"
#include "Token.hpp"

#include <vector>

namespace Algebra
{

  class Parser
  {

  public:
    static ExpressionPtr parse(
        const std::string &expression);

  private:
    explicit Parser(
        std::vector<Token> tokens);

    ExpressionPtr parseExpression();

    ExpressionPtr parseTerm();

    ExpressionPtr parseFactor();

    ExpressionPtr parsePrimary();

    bool match(TokenType type);

    const Token &peek() const;

    Token consume(
        TokenType type);

  private:
    std::vector<Token> tokens_;

    std::size_t position_ = 0;
  };

}