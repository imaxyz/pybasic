import graphene


class SampleModel(graphene.ObjectType):
    """サンプルのモデル"""

    hello = graphene.String(name=graphene.String(default_value="World"))
    hoge = graphene.String(name=graphene.String(default_value="Hoge"))
    goodbye = graphene.String()

    def resolve_hello(self, info, name):
        return 'Hello ' + name

    def resolve_hoge(self, info, name):
        return 'Hello ' + name

    def resolve_goodbye(self, info):
        return 'Goood bye!'


# スキーマを取得
schema = graphene.Schema(query=SampleModel)

# スキーマに対してクエリする
result = schema.execute('{ hello }')
result2 = schema.execute('{ hoge }')
result3 = schema.execute('{ goodbye }')

print(result, result2, result3, sep="\n")
