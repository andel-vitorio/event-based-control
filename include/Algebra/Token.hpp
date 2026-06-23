#pragma once

#include <string>

namespace Algebra
{

  enum class TokenType
  {
    Number,
    Identifier,

    Plus,
    Minus,
    Multiply,
    Divide,
    Power,

    LeftParen,
    RightParen,

    Comma,

    End
  };

  struct Token
  {

    TokenType type;

    std::string text;
  };

}