from context import context
class A:
    def __init__(self, context: context):
            self.context = context

    def _run(self, context:str = "hello"):
        print("Running class A with value:", self.context.user)
        self.context.info = "Class A has run."
        return context