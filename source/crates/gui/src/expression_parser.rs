//! This is a simple parser for math expressions so that users can enter them into the GUI's slider numeric inputs.
//!
//! It evaluates all expressions as f64. It's a pretty standard Pratt parser, with operator precedence + associativity
//! matching what you'd expect from most programming languages (PEMDAS, operators are left-associative except for
//! exponentiation).
//!
//! The one deviation from C-like expression languages is that the `%` operator is not an infix modulo operator, but a
//! postfix "percentage" operator that means "divide by 100". This is a lot more useful for our purposes, where some
//! effect parameters are logically percentages.

use std::{
    fmt::{self, Display, Write},
    num::ParseFloatError,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Token {
    Number(f64),
    Plus,
    Minus,
    Multiply,
    Divide,
    Power,
    Percent,
    LParen,
    RParen,
}

impl Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Number(n) => n.fmt(f),
            Self::Plus => f.write_char('+'),
            Self::Minus => f.write_char('-'),
            Self::Multiply => f.write_char('*'),
            Self::Divide => f.write_char('/'),
            Self::Power => f.write_str("**"),
            Self::Percent => f.write_char('%'),
            Self::LParen => f.write_char('('),
            Self::RParen => f.write_char(')'),
        }
    }
}

#[derive(Debug, Default, PartialEq, Clone)]
pub enum ParseError {
    UnexpectedChar(char),
    UnexpectedToken(Token),
    UnexpectedEOF,
    ExpectedChar(char),
    ExpectedOperator,
    InvalidLHS,
    InvalidNumber(ParseFloatError),
    #[default]
    Other,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedChar(c) => write!(f, "Unexpected character: {c}"),
            Self::UnexpectedToken(t) => write!(f, "Unexpected token: {t}"),
            Self::UnexpectedEOF => write!(f, "Unexpected EOF"),
            Self::ExpectedChar(c) => write!(f, "Expected a \"{c}\" character"),
            Self::ExpectedOperator => write!(f, "Expected an operator"),
            Self::InvalidLHS => write!(f, "Invalid left-hand side"),
            Self::InvalidNumber(e) => write!(f, "Invalid number: {e}"),
            Self::Other => write!(f, "Unknown error"),
        }
    }
}

impl From<ParseFloatError> for ParseError {
    fn from(value: ParseFloatError) -> Self {
        Self::InvalidNumber(value)
    }
}

impl std::error::Error for ParseError {}

struct Lexer<'a> {
    remaining: &'a str,
    cur: Option<Token>,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Result<Self, ParseError> {
        let mut lexer = Self {
            remaining: input,
            cur: None,
        };
        lexer.advance()?;
        Ok(lexer)
    }

    fn peek_next(&self) -> Option<(char, &'a str)> {
        let mut chars = self.remaining.chars();
        Some((chars.next()?, chars.as_str()))
    }

    fn lex_digit_string(&mut self) -> usize {
        let mut len = 0;
        while let Some(('0'..='9', rest)) = self.peek_next() {
            len += 1;
            self.remaining = rest;
        }
        len
    }

    fn unexpected(&self) -> ParseError {
        match self.peek_next() {
            Some((c, _)) => ParseError::UnexpectedChar(c),
            None => ParseError::UnexpectedEOF,
        }
    }

    fn lex_number(&mut self) -> Result<f64, ParseError> {
        let rest = self.remaining;

        if let Some(('.', rest)) = self.peek_next() {
            // Leading decimal point followed by a nonzero number of digits
            self.remaining = rest;

            if self.lex_digit_string() == 0 {
                return Err(self.unexpected());
            }
        } else {
            // Nonzerro number of digits followed by a decimal point and optional string of digits
            if self.lex_digit_string() == 0 {
                return Err(self.unexpected());
            }
            if let Some(('.', rest)) = self.peek_next() {
                self.remaining = rest;
            }
            self.lex_digit_string();
        }

        // Exponent ([eE] [+-]? [0-9]+)
        if let Some(('e' | 'E', rest)) = self.peek_next() {
            self.remaining = rest;

            // The exponent may be followed by a plus or minus
            if let Some(('+' | '-', rest)) = self.peek_next() {
                self.remaining = rest;
            }

            if self.lex_digit_string() == 0 {
                return Err(self.unexpected());
            }
        }

        let num_str = &rest[..rest.len() - self.remaining.len()];
        let num = num_str.parse::<f64>()?;
        Ok(num)
    }

    fn advance(&mut self) -> Result<(), ParseError> {
        let next_token = loop {
            let Some((peeked, rest)) = self.peek_next() else {
                break None;
            };
            match peeked {
                // Operators
                '+' => {
                    break {
                        self.remaining = rest;
                        Some(Token::Plus)
                    };
                }
                '-' => {
                    break {
                        self.remaining = rest;
                        Some(Token::Minus)
                    };
                }
                '/' => {
                    break {
                        self.remaining = rest;
                        Some(Token::Divide)
                    };
                }
                '%' => {
                    break {
                        self.remaining = rest;
                        Some(Token::Percent)
                    };
                }
                '(' => {
                    break {
                        self.remaining = rest;
                        Some(Token::LParen)
                    };
                }
                ')' => {
                    break {
                        self.remaining = rest;
                        Some(Token::RParen)
                    };
                }

                '*' => {
                    break {
                        self.remaining = rest;
                        if let Some(('*', rest)) = self.peek_next() {
                            self.remaining = rest;
                            Some(Token::Power)
                        } else {
                            Some(Token::Multiply)
                        }
                    };
                }

                '0'..='9' | '.' => {
                    break Some(Token::Number(self.lex_number()?));
                }

                // Whitespace
                ' ' | '\r' | '\n' | '\t' | '\x0C' => {
                    self.remaining = rest;
                    continue;
                }

                // Unexpected character
                c => return Err(ParseError::UnexpectedChar(c)),
            }
        };

        self.cur = next_token;
        Ok(())
    }
}

fn prefix_binding_power(op: &Token) -> usize {
    match op {
        Token::Plus | Token::Minus => 7,
        _ => panic!("not a prefix operator: {op:?}"),
    }
}

fn postfix_binding_power(op: &Token) -> Option<usize> {
    match op {
        Token::Percent => Some(9),
        _ => None,
    }
}

fn infix_binding_power(op: &Token) -> Option<(usize, usize)> {
    match op {
        Token::Plus | Token::Minus => Some((1, 2)),
        Token::Multiply | Token::Divide => Some((3, 4)),
        Token::Power => Some((6, 5)),
        _ => None,
    }
}

fn eval_expr(lexer: &mut Lexer, min_binding_power: usize) -> Result<f64, ParseError> {
    let mut lhs = match lexer.cur {
        Some(Token::LParen) => {
            lexer.advance()?;
            let res = eval_expr(lexer, 0)?;
            if lexer.cur != Some(Token::RParen) {
                return Err(ParseError::ExpectedChar(')'));
            }
            lexer.advance()?;
            Ok(res)
        }
        Some(Token::Number(value)) => {
            lexer.advance()?;
            Ok(value)
        }
        Some(Token::Plus) => {
            lexer.advance()?;
            let inner_value = eval_expr(lexer, prefix_binding_power(&Token::Plus))?;
            // unary plus does nothing
            Ok(inner_value)
        }
        Some(Token::Minus) => {
            lexer.advance()?;
            let inner_value = eval_expr(lexer, prefix_binding_power(&Token::Minus))?;
            // unary negation
            Ok(-inner_value)
        }
        _ => Err(ParseError::InvalidLHS),
    }?;

    loop {
        let op = match lexer.cur {
            Some(token) => match token {
                Token::Plus
                | Token::Minus
                | Token::Multiply
                | Token::Divide
                | Token::Power
                | Token::Percent => Ok(token),
                Token::RParen => break Ok(lhs),
                _ => Err(ParseError::ExpectedOperator),
            },
            None => break Ok(lhs),
        }?;

        if let Some(left_bp) = postfix_binding_power(&op) {
            if left_bp < min_binding_power {
                break Ok(lhs);
            }
            lexer.advance()?;
            lhs = match op {
                Token::Percent => lhs * 0.01,
                _ => panic!("unhandled op: {op:?}"),
            };
            continue;
        }

        if let Some((left_bp, right_bp)) = infix_binding_power(&op) {
            if left_bp < min_binding_power {
                break Ok(lhs);
            }

            lexer.advance()?;
            let rhs = eval_expr(lexer, right_bp)?;

            lhs = match op {
                Token::Plus => lhs + rhs,
                Token::Minus => lhs - rhs,
                Token::Multiply => lhs * rhs,
                Token::Divide => lhs / rhs,
                Token::Power => lhs.powf(rhs),
                _ => panic!("unhandled op: {op:?}"),
            };
            continue;
        }

        panic!("unhandled op: {op:?}");
    }
}

pub fn eval_expression_string(string: &str) -> Result<f64, ParseError> {
    let mut lexer = Lexer::new(string)?;

    let result = eval_expr(&mut lexer, 0)?;
    if let Some(cur) = lexer.cur {
        return Err(ParseError::UnexpectedToken(cur));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(s: &str) -> f64 {
        eval_expression_string(s).expect("expected expression to parse")
    }

    fn err(s: &str) -> ParseError {
        eval_expression_string(s).expect_err("expected expression to fail to parse")
    }

    // ---------- Numbers ----------

    #[test]
    fn parses_integer() {
        assert_eq!(eval("42"), 42.0);
    }

    #[test]
    fn parses_zero() {
        assert_eq!(eval("0"), 0.0);
    }

    #[test]
    fn parses_decimal() {
        assert_eq!(eval("1.25"), 1.25);
    }

    #[test]
    fn parses_decimal_leading_dot() {
        assert_eq!(eval(".5"), 0.5);
    }

    #[test]
    fn parses_decimal_trailing_dot() {
        assert_eq!(eval("5."), 5.0);
    }

    #[test]
    fn parses_scientific_notation_positive_exponent() {
        assert_eq!(eval("1e+5"), 100_000.0);
    }

    #[test]
    fn parses_scientific_notation_negative_exponent() {
        assert_eq!(eval("1e-5"), 0.000_01);
    }

    #[test]
    fn parses_decimal_with_scientific_notation() {
        assert_eq!(eval("1.5e+2"), 150.0);
    }

    #[test]
    fn uppercase_e_in_scientific_notation() {
        assert_eq!(eval("1E+5"), 100_000.0);
    }

    #[test]
    fn unsigned_exponent_is_valid() {
        assert_eq!(eval("2e5"), 200_000.0);
    }

    // ---------- Individual operators ----------

    #[test]
    fn addition() {
        assert_eq!(eval("1 + 2"), 3.0);
    }

    #[test]
    fn subtraction() {
        assert_eq!(eval("5 - 3"), 2.0);
    }

    #[test]
    fn multiplication() {
        assert_eq!(eval("4 * 3"), 12.0);
    }

    #[test]
    fn division() {
        assert_eq!(eval("10 / 4"), 2.5);
    }

    #[test]
    fn power() {
        assert_eq!(eval("2 ** 10"), 1024.0);
    }

    #[test]
    fn power_fractional_exponent() {
        assert!((eval("9 ** 0.5") - 3.0).abs() < 1e-12);
    }

    #[test]
    fn percent_postfix() {
        assert_eq!(eval("50%"), 0.5);
    }

    #[test]
    fn percent_chained() {
        // 5% = 0.05, then 0.05% = 0.0005
        assert_eq!(eval("5%%"), 0.0005);
    }

    // ---------- Unary operators ----------

    #[test]
    fn unary_plus() {
        assert_eq!(eval("+5"), 5.0);
    }

    #[test]
    fn unary_minus() {
        assert_eq!(eval("-5"), -5.0);
    }

    #[test]
    fn double_negation() {
        assert_eq!(eval("--5"), 5.0);
    }

    #[test]
    fn mixed_unary_signs() {
        assert_eq!(eval("-+-+5"), 5.0);
    }

    #[test]
    fn unary_minus_on_decimal() {
        assert_eq!(eval("-3.5"), -3.5);
    }

    #[test]
    fn binary_op_followed_by_unary() {
        // `1 + + 2` -> 1 + (+2) = 3
        assert_eq!(eval("1 + + 2"), 3.0);
    }

    #[test]
    fn binary_op_followed_by_unary_minus() {
        // `1 - -2` -> 1 - (-2) = 3
        assert_eq!(eval("1 - -2"), 3.0);
    }

    // ---------- Precedence ----------

    #[test]
    fn multiplication_binds_tighter_than_addition() {
        assert_eq!(eval("1 + 2 * 3"), 7.0);
        assert_eq!(eval("2 * 3 + 1"), 7.0);
    }

    #[test]
    fn division_binds_tighter_than_subtraction() {
        assert_eq!(eval("10 - 6 / 2"), 7.0);
    }

    #[test]
    fn power_binds_tighter_than_multiplication() {
        assert_eq!(eval("2 * 3 ** 2"), 18.0);
    }

    #[test]
    fn power_binds_tighter_than_addition() {
        assert_eq!(eval("1 + 2 ** 3"), 9.0);
    }

    #[test]
    fn unary_minus_binds_tighter_than_power() {
        // Prefix minus has binding power 7, `**` has (6, 5).
        // So `-2 ** 2` parses as `(-2) ** 2` = 4.
        assert_eq!(eval("-2 ** 2"), 4.0);
    }

    #[test]
    fn percent_binds_tighter_than_addition() {
        assert_eq!(eval("100 + 50%"), 100.5);
    }

    #[test]
    fn percent_binds_tighter_than_multiplication() {
        assert_eq!(eval("2 * 50%"), 1.0);
    }

    #[test]
    fn percent_binds_tighter_than_power() {
        // `2 ** 50%` -> `2 ** 0.5` = sqrt(2)
        assert!((eval("2 ** 50%") - 2f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn percent_binds_tighter_than_unary_minus() {
        // Postfix `%` (bp 9) is tighter than prefix `-` (bp 7).
        // So `-50%` is `-(50%)` = -0.5.
        assert_eq!(eval("-50%"), -0.5);
    }

    #[test]
    fn precedence_combined() {
        // 1 + 2 * 3 ** 2 = 1 + 2 * 9 = 19
        assert_eq!(eval("1 + 2 * 3 ** 2"), 19.0);
    }

    // ---------- Associativity ----------

    #[test]
    fn subtraction_is_left_associative() {
        // (10 - 3) - 2 = 5, not 10 - (3 - 2) = 9
        assert_eq!(eval("10 - 3 - 2"), 5.0);
    }

    #[test]
    fn division_is_left_associative() {
        // (16 / 4) / 2 = 2, not 16 / (4 / 2) = 8
        assert_eq!(eval("16 / 4 / 2"), 2.0);
    }

    #[test]
    fn addition_is_left_associative() {
        assert_eq!(eval("1 + 2 + 3"), 6.0);
    }

    #[test]
    fn multiplication_is_left_associative() {
        assert_eq!(eval("2 * 3 * 4"), 24.0);
    }

    #[test]
    fn power_is_right_associative() {
        // 2 ** (3 ** 2) = 2 ** 9 = 512, not (2 ** 3) ** 2 = 64
        assert_eq!(eval("2 ** 3 ** 2"), 512.0);
    }

    // ---------- Parentheses ----------

    #[test]
    fn simple_parentheses() {
        assert_eq!(eval("(1 + 2)"), 3.0);
    }

    #[test]
    fn parentheses_override_precedence() {
        assert_eq!(eval("(1 + 2) * 3"), 9.0);
    }

    #[test]
    fn parentheses_on_right_side() {
        assert_eq!(eval("3 * (1 + 2)"), 9.0);
    }

    #[test]
    fn nested_parentheses() {
        assert_eq!(eval("((1 + 2) * (3 + 4))"), 21.0);
    }

    #[test]
    fn parentheses_with_unary_minus() {
        assert_eq!(eval("-(1 + 2)"), -3.0);
    }

    #[test]
    fn parenthesized_power_left_associative() {
        // (2 ** 3) ** 2 = 64, contrasting with right-associativity test above.
        assert_eq!(eval("(2 ** 3) ** 2"), 64.0);
    }

    #[test]
    fn parentheses_with_percent() {
        // (1 + 1)% = 2% = 0.02
        assert_eq!(eval("(1 + 1)%"), 0.02);
    }

    // ---------- Whitespace ----------

    #[test]
    fn leading_and_trailing_whitespace() {
        assert_eq!(eval("   1 + 2   "), 3.0);
    }

    #[test]
    fn whitespace_between_all_tokens() {
        assert_eq!(eval(" ( 1 + 2 ) * 3 "), 9.0);
    }

    #[test]
    fn no_whitespace_required() {
        assert_eq!(eval("1+2*3"), 7.0);
    }

    #[test]
    fn tabs_and_newlines_skipped() {
        assert_eq!(eval("1\t+\n2"), 3.0);
    }

    // ---------- Errors ----------

    #[test]
    fn empty_string_is_invalid_lhs() {
        assert_eq!(err(""), ParseError::InvalidLHS);
    }

    #[test]
    fn whitespace_only_is_invalid_lhs() {
        assert_eq!(err("   "), ParseError::InvalidLHS);
    }

    #[test]
    fn lone_operator_is_invalid_lhs() {
        // `*` cannot start an expression.
        assert_eq!(err("*"), ParseError::InvalidLHS);
    }

    #[test]
    fn lone_close_paren_is_invalid_lhs() {
        assert_eq!(err(")"), ParseError::InvalidLHS);
    }

    #[test]
    fn empty_parentheses_are_invalid_lhs() {
        assert_eq!(err("()"), ParseError::InvalidLHS);
    }

    #[test]
    fn trailing_binary_operator_is_invalid_lhs() {
        assert_eq!(err("1 +"), ParseError::InvalidLHS);
    }

    #[test]
    fn trailing_binary_operator_after_expression() {
        assert_eq!(err("1 + 2 *"), ParseError::InvalidLHS);
    }

    #[test]
    fn unary_with_no_operand_is_invalid_lhs() {
        assert_eq!(err("-"), ParseError::InvalidLHS);
    }

    #[test]
    fn unopened_paren_at_end() {
        assert_eq!(err("1 + 2)"), ParseError::UnexpectedToken(Token::RParen));
    }

    #[test]
    fn unopened_paren_in_middle() {
        assert_eq!(err("1)1"), ParseError::UnexpectedToken(Token::RParen));
    }

    #[test]
    fn unopened_paren_complex() {
        assert_eq!(err("1+1)+1"), ParseError::UnexpectedToken(Token::RParen));
    }

    #[test]
    fn unclosed_paren_expects_close() {
        assert_eq!(err("(1 + 2"), ParseError::ExpectedChar(')'));
    }

    #[test]
    fn unclosed_outer_paren_with_inner_close() {
        assert_eq!(err("((1 + 2)"), ParseError::ExpectedChar(')'));
    }

    #[test]
    fn lone_open_paren_is_invalid_lhs() {
        // The inner expression is empty, so the LHS check trips first.
        assert_eq!(err("("), ParseError::InvalidLHS);
    }

    #[test]
    fn two_numbers_in_a_row_expect_operator() {
        assert_eq!(err("1 2"), ParseError::ExpectedOperator);
    }

    #[test]
    fn number_followed_by_open_paren_expects_operator() {
        // No implicit multiplication.
        assert_eq!(err("2(3)"), ParseError::ExpectedOperator);
    }

    #[test]
    fn unrecognized_character_at_start() {
        assert_eq!(err("a"), ParseError::UnexpectedChar('a'));
    }

    #[test]
    fn unrecognized_character_mid_expression() {
        assert_eq!(err("1 + a"), ParseError::UnexpectedChar('a'));
    }

    // ---------- IEEE 754 edge cases ----------

    #[test]
    fn division_by_zero_yields_infinity() {
        assert!(eval("1 / 0").is_infinite());
        assert!(eval("1 / 0").is_sign_positive());
    }

    #[test]
    fn negative_division_by_zero_yields_negative_infinity() {
        let v = eval("-1 / 0");
        assert!(v.is_infinite() && v.is_sign_negative());
    }

    #[test]
    fn zero_divided_by_zero_is_nan() {
        assert!(eval("0 / 0").is_nan());
    }
}
