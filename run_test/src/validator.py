class Validator():
    def __init__(self, data_format: dict, verbose=False) -> None:
        assert isinstance(data_format, dict)
        self.data_format = data_format
        assert {'keys', 'dtype', 'k'}.issubset(set(self.data_format.keys()))
        if verbose:
            print('\nValidation details:')
            for k, v in self.data_format.items():
                print('  {}: {}'.format(k, v))
        self.data = None
        print('\nValidation:')

    def check_data(self, result) -> None:
        raise NotImplementedError


    def check_samples(self, result) -> None:
        raise NotImplementedError


    def check_dtype(self, result) -> None:
        raise NotImplementedError


    def check_keys(self, result) -> None:
        raise NotImplementedError


    def check_length(self, result) -> None:
        msg = '  Checking list length...'
        assert isinstance(result, list)
        if len(result) > self.data_format['k']:
            raise MaximumExceedError("The number of predictions must not exceed {}. you requested {}.".format(self.data_format['k'], len(result)))
        print(msg+' Done')
        pass


    def validate(self, result) -> None:
        for k, res in result.items():
            self.check_length(res)
            for p in res:
                self.check_data(p)
                self.check_keys(p)
                self.check_dtype(p)
                self.check_details(p)


    def get_data(self) -> None:
        return self.data


class DictValidator(Validator):
    def check_data(self, result) -> None:
        msg = '  Checking data...'
        print(msg, end='\r')
        assert isinstance(result, dict)
        print(msg+' Done')


    def check_samples(self, result) -> None:
        msg = '  Checking samples...'
        print(msg, end='\r')
        if set(result.keys()) != self.data_format['samples']:
            raise SampleError('Missing samples or invalid samples found.')
        print(msg+' Done')


    def check_dtype(self, result) -> None:
        msg = '  Checking dtype...'
        print(msg, end='\r')
        for k, v in result.items():
            if not isinstance(v, self.data_format['dtype'][k]):
                raise DtypeError('Invalid data type found in index {}: {}!={}(expected)'.format(k, type(v), self.data_format['dtype'][k]))
        print(msg+' Done')


    def check_keys(self, result) -> None:
        msg = '  Checking keys...'
        print(msg, end='\r')
        if set(result.keys()) != set(self.data_format['keys']):
            raise ElementError('Invalid key found in  {}. Must be in {}'.format(set(result.keys()), self.data_format['keys']))
        print(msg+' Done')


    def check_details(self, result) -> None:
        if result["category_id"] not in self.data_format["categories"]:
            raise ClassError('Invalid class {}. Must be in {}'.format(result["category_id"], self.data_format["categories"]))
        pass

class SampleError(Exception):
    pass


class ElementError(Exception):
    pass


class DtypeError(Exception):
    pass


class ExtentionError(Exception):
    pass


class DelimiterError(Exception):
    pass


class NumColumnsError(Exception):
    pass


class NullError(Exception):
    pass


class DiscreteDataError(Exception):
    pass


class MaximumExceedError(Exception):
    pass


class InstanceError(Exception):
    pass


class ClassError(Exception):
    pass