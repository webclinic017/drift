import argparse
from ast import Expression, literal_eval
from typing import Union

import libcst as cst
from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand
from libcst.codemod.visitors import AddImportsVisitor


class ReplaceFunctionCommand(VisitorBasedCodemodCommand):

    # Add a description so that future codemodders can see what this does.
    DESCRIPTION: str = "Replaces the body of a function with pass."

    def __init__(self, context: CodemodContext) -> None:
        # Initialize the base class with context, and save our args. Remember, the
        # "dest" for each argument we added above must match a parameter name in
        # this init.
        super().__init__(context)

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        functions_docstring = updated_node.get_docstring()
        docstring_should_be = '"""No docstring here yet."""'
        if functions_docstring is not None:
            docstring_should_be = '"""\n{}\n\n"""'.format(functions_docstring)

        replace_function = cst.FunctionDef(
            name=updated_node.name,
            params=updated_node.params,  # cst.Parameters(),
            body=cst.IndentedBlock(
                body=[
                    cst.SimpleStatementLine(
                        body=[
                            cst.Expr(
                                value=cst.SimpleString(
                                    value=docstring_should_be,
                                    lpar=[],
                                    rpar=[],
                                ),
                                semicolon=cst.MaybeSentinel.DEFAULT,
                            ),
                        ],
                        leading_lines=[],
                        trailing_whitespace=cst.TrailingWhitespace(
                            whitespace=cst.SimpleWhitespace(
                                value="",
                            ),
                            comment=None,
                            newline=cst.Newline(
                                value=None,
                            ),
                        ),
                    ),
                    cst.SimpleStatementLine(
                        body=[
                            cst.Pass(),
                        ],
                    ),
                ]
            ),
        )

        return replace_function

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        new_body = []
        for body_item in updated_node.body.body:
            if type(body_item) is cst.FunctionDef:
                new_body.append(self.leave_FunctionDef(body_item, body_item))

        return updated_node.with_changes(body=cst.IndentedBlock(new_body))

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        new_module_body = []
        for node in original_node.body:
            if type(node) is cst.FunctionDef:
                new_module_body.append(self.leave_FunctionDef(node, node))

            if type(node) is cst.ClassDef:
                new_module_body.append(self.leave_ClassDef(node, node))

        replace_function = cst.Module(body=new_module_body)

        return replace_function
