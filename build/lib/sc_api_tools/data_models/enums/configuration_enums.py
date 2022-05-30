from enum import Enum


class ConfigurationEntityType(Enum):
    HYPER_PARAMETER_GROUP = 'HYPER_PARAMETER_GROUP'
    COMPONENT_PARAMETERS = 'COMPONENT_PARAMETERS'

    def __str__(self):
        """
        Returns the string representation of the ConfigurationEntityType instance
        :return:
        """
        return self.value


class ParameterDataType(Enum):
    BOOLEAN = 'boolean'
    FLOAT = 'float'
    STRING = 'string'
    INTEGER = 'integer'

    def __str__(self):
        """
        Returns the string representation of the ParameterDataType instance
        :return:
        """
        return self.value


class ParameterInputType(Enum):
    INPUT = 'input'
    SELECTABLE = 'selectable'

    def __str__(self):
        """
        Returns the string representation of the ParameterInputType instance
        :return:
        """
        return self.value


class ConfigurableParameterType(Enum):
    CONFIGURABLE_PARAMETERS = 'CONFIGURABLE_PARAMETERS'
    PARAMETER_GROUP = 'PARAMETER_GROUP'

    def __str__(self):
        """
        Returns the string representation of the ConfigurableParameterType instance
        :return:
        """
        return self.value
