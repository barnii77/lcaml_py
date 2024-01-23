import unittest
from lexer import Lexer, Syntax, TokenKind, LexError, Token


class TestLexer(unittest.TestCase):
    def setUp(self):
        self.syntax = Syntax()

    def test_empty_code(self):
        lexer = Lexer("", self.syntax)
        tokens = lexer()
        self.assertEqual(tokens, [])

    def test_let_keyword(self):
        lexer = Lexer("let ", self.syntax)
        tokens = lexer()
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenKind.LET)
        self.assertEqual(tokens[0], "let")

    def test_identifier(self):
        lexer = Lexer("variableName", self.syntax)
        tokens = lexer()
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenKind.IDENTIFIER)
        self.assertEqual(tokens[0], "variableName")

    def test_integer(self):
        lexer = Lexer("12345", self.syntax)
        tokens = lexer()
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenKind.INTEGER)
        self.assertEqual(tokens[0], "12345")

    def test_floating_point(self):
        lexer = Lexer("123.45", self.syntax)
        tokens = lexer()
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenKind.FLOATING_POINT)
        self.assertEqual(tokens[0], "123.45")

    def test_string_literal(self):
        lexer = Lexer('"Hello, World!"', self.syntax)
        tokens = lexer()
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenKind.STRING_LITERAL)
        self.assertEqual(tokens[0], '"Hello, World!"')

    def test_equals(self):
        lexer = Lexer("=", self.syntax)
        tokens = lexer()
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenKind.EQUALS)
        self.assertEqual(tokens[0], "=")

    def test_semicolon(self):
        lexer = Lexer(";", self.syntax)
        tokens = lexer()
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenKind.SEMICOLON)
        self.assertEqual(tokens[0], ";")

    def test_comment(self):
        lexer = Lexer("-- This is a comment\n", self.syntax)
        tokens = lexer()
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenKind.COMMENT)
        self.assertTrue(tokens[0].value.startswith("--"))

    def test_operator(self):
        lexer = Lexer("+", self.syntax)
        tokens = lexer()
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenKind.OPERATOR)
        self.assertEqual(tokens[0], "+")

    def test_complex_code(self):
        code = """
        let x = 10; -- x y z
        let y = 20;
        let z = x + y;
        """
        lexer = Lexer(code, self.syntax)
        tokens = lexer()
        expected_tokens = [
            Token(TokenKind.LET, "let"),
            Token(TokenKind.IDENTIFIER, "x"),
            Token(TokenKind.EQUALS, "="),
            Token(TokenKind.INTEGER, "10"),
            Token(TokenKind.SEMICOLON, ";"),
            Token(TokenKind.COMMENT, "-- x y z"),
            Token(TokenKind.LET, "let"),
            Token(TokenKind.IDENTIFIER, "y"),
            Token(TokenKind.EQUALS, "="),
            Token(TokenKind.INTEGER, "20"),
            Token(TokenKind.SEMICOLON, ";"),
            Token(TokenKind.LET, "let"),
            Token(TokenKind.IDENTIFIER, "z"),
            Token(TokenKind.EQUALS, "="),
            Token(TokenKind.IDENTIFIER, "x"),
            Token(TokenKind.OPERATOR, "+"),
            Token(TokenKind.IDENTIFIER, "y"),
            Token(TokenKind.SEMICOLON, ";"),
        ]
        self.assertEqual(tokens, expected_tokens)

    def test_lex_error(self):
        lexer = Lexer("?/0x", self.syntax)
        with self.assertRaises(LexError):
            lexer()


if __name__ == "__main__":
    unittest.main()
