name_dict = {
            "车辆名称": "carName",
            "车型报价": "carPrice",
            "上牌时间": "carYear",
            "表显里程": "carMileage",
            "所在地": "carLocation",
            "车源网址": "carInfoUrl",
            "车身颜色": "carColor",
            "car_seriesid": "carSeriesId",
            "car_infoid": "carInfoId",
            "car_specid": "carSpecid",
            "热度指数": "carHeat",
            "关注指数": "carFocusHeat",
            "咨询指数": "carConsultHeat",
            "搜索指数": "carSearchHeat",
            "配置亮点": "carHighlight",
            "厂商": "carManufacturer",
            "车辆级别": "carLevel",
            "能源类型": "carEnergyType",
            "环保标准": "carEnvProtStandard",
            "上市时间": "carMarketTime",
            "最高车速(km/h)": "carMaxSpeed",
            "官方0-100km/h加速(s)": "carAccelerationTime",
            "WLTC综合油耗(L/100km)": "carWLTCOil",
            "NEDC综合油耗(L/100km)": "carNEDCOil",
            "长度(mm)": "carLong",
            "宽度(mm)": "carWidth",
            "高度(mm)": "carHeight",
            "车门数(个)": "carDoorNum",
            "座位数(个)": "carSeatNum",
            "后备厢容积(L)": "carVOfTrunk",
            "整备质量(kg)": "carWeight",
            "最大满载质量(kg)": "carLoadWeight",
            "排量(mL)": "carDisplacement",
            "气缸数(个)": "carCylinderNum",
            "每缸气门数(个)": "carValveNumPCyl",
            "最大马力(Ps)": "carMaxHorsepower",
            "最大功率(kW)": "carMaxPower",
            "最大功率转速(rpm)": "carMaxPowerRPM",
            "最大扭矩(N·m)": "carMaxTorque",
            "最大扭矩转速(rpm)": "carMaxTorqueRPM",
            "最大净功率(kW)": "carMaxNetPower",
            "燃油标号": "carOilLabel",
            "缸盖材料": "carCylHeadMaterial",
            "缸体材料": "carCylBodyMaterial",
            "电动机总功率(kW)": "carElecPower",
            "电动机总扭矩(N·m)": "carElecTorque",
            "前电动机最大功率(kW)": "carFrontElecMaxPower",
            "前电动机最大扭矩(N·m)": "carFrontElecMaxTorque",
            "驱动电机数": "carElecMatNum",
            "电池冷却方式": "carBatteryCooling",
            "CLTC纯电续航里程(km)": "carCLTCRange",
            "NEDC纯电续航里程(km)": "carNEDCRange",
            "电池能量(kWh)": "carVOfBattery",
            "电池能量密度(Wh/kg)": "carBatteryDensity",
            "百公里耗电量(kWh/100km)": "carElecConsumption100",
            "快充功能": "carIsFastCharge",
            "挡位个数": "carGearNum",
            "变速箱类型": "carGearType",
            "驱动方式": "carDriveType",
            "前悬架类型": "carFrontSuspension",
            "后悬架类型": "carRearSuspension",
            "驻车制动类型": "carParkBrakeType",
            "前轮胎规格": "carFrontTireSize",
            "后轮胎规格": "carRearTireSize",
            "备胎规格": "carSpareTireSize"
        }
class CarInfo:
    def __init__(self):
        self.carName = ""
        self.carPrice = 0.0
        self.carYear = ""
        self.carMileage = 0.0
        self.carLocation = ""
        self.carInfoUrl = ""
        self.carColor = ""
        self.carSeriesId = ""
        self.carInfoId = ""
        self.carSpecid = ""
        self.carHeat = 0.0
        self.carFocusHeat = 0.0
        self.carConsultHeat = 0.0
        self.carSearchHeat = 0.0
        self.carHighlight = ""
        self.carManufacturer = ""
        self.carLevel = ""
        self.carEnergyType = ""
        self.carEnvProtStandard = ""
        self.carMarketTime = ""
        self.carMaxSpeed = 0
        self.carAccelerationTime = 0.0
        self.carWLTCOil = 0.0
        self.carNEDCOil = 0.0
        self.carLong = 0
        self.carWidth = 0
        self.carHeight = 0
        self.carDoorNum = 0
        self.carSeatNum = 0
        self.carVOfTrunk = ""
        self.carWeight = 0
        self.carLoadWeight = 0
        self.carDisplacement = 0
        self.carCylinderNum = 0
        self.carValveNumPCyl = 0
        self.carMaxHorsepower = 0
        self.carMaxPower = 0
        self.carMaxPowerRPM = ""
        self.carMaxTorque = 0
        self.carMaxTorqueRPM = ""
        self.carMaxNetPower = 0
        self.carOilLabel = ""
        self.carCylHeadMaterial = ""
        self.carCylBodyMaterial = ""
        self.carElecPower = 0
        self.carElecTorque = 0
        self.carFrontElecMaxPower = 0
        self.carFrontElecMaxTorque = 0
        self.carElecMatNum = ""
        self.carBatteryCooling = ""
        self.carCLTCRange = ""
        self.carNEDCRange = ""
        self.carVOfBattery = 0
        self.carBatteryDensity = ""
        self.carElecConsumption100 = 0.0
        self.carIsFastCharge = 0
        self.carGearNum = 0
        self.carGearType = ""
        self.carDriveType = ""
        self.carFrontSuspension = ""
        self.carRearSuspension = ""
        self.carParkBrakeType = ""
        self.carFrontTireSize = ""
        self.carRearTireSize = ""
        self.carSpareTireSize = ""

    def setValue(self, AttibuteName, AttributeValue):
        if AttibuteName in name_dict:
            setattr(self, name_dict[AttibuteName], AttributeValue)
            return True
        else:
            print("Error: " + AttibuteName + " is not in name_dict!")
            return False

    def getValue(self, AttibuteName):
        if AttibuteName in name_dict:
            return getattr(self, name_dict[AttibuteName])
        else:
            print("Error: " + AttibuteName + " is not in name_dict!")
            return None

    @staticmethod
    def getNameDict():
        return name_dict
