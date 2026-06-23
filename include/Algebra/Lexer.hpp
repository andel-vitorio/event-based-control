#pragma once

#include "Token.hpp"

#include <string>
#include <vector>

namespace Algebra
{

  class Lexer
  {

  public:
    explicit Lexer(
        const std::string &expression);

    std::vector<Token> tokenize();

  private:
    std::string text_;

    size_t position_ = 0;

    char current() const;

    void advance();

    void skipSpaces();

    Token number();

    Token identifier();
  };

}