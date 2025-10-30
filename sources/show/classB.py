from context import context
class B:
    def __init__(self, context: context):
            self.context = context

    def _run(self, context:str = "hello"):
        print("Running class B with value:", self.context.info)
        return context