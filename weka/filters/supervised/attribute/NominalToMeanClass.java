/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    NominalToMeanClass.java
 *    Copyright (C) 2018 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.filters.supervised.attribute;

import java.util.ArrayList;

import weka.core.*;
import weka.core.Capabilities.*;
import weka.filters.*;

public class NominalToMeanClass extends SimpleBatchFilter implements SupervisedFilter {

    protected Range m_SelectedCols = new Range("first-last");

    protected int[] m_SelectedAttributes;

    protected boolean[] m_AttToBeModified;

    protected double[][] m_Codes;

    public void setAttributeIndices(String range) {
        m_SelectedCols.setRanges(range);
    }

    public String getAttributeIndices() {
        return m_SelectedCols.getRanges();
    }

    public String globalInfo() {
        return "Performs mean encoding for all specified nominal attributes.";
    }

    public String attributeIndicesTipText() {
        return "The attriutes for which mean encoding should be performed.\n\n"
            + "Numeric attributes and the class attribute are ignored.";
    }

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.enableAllAttributes();
        result.enable(Capability.BINARY_CLASS);
        result.enable(Capability.NUMERIC_CLASS);

        return result;
    }

    protected Instances determineOutputFormat(Instances inputFormat) {
        
        m_SelectedCols.setUpper(inputFormat.numAttributes() - 1);
        m_SelectedAttributes = m_SelectedCols.getSelection();

        ArrayList<Attribute> newAtts = new ArrayList<Attribute>();

        for(int attIdx : m_SelectedAttributes) {
            Attribute att = inputFormat.attribute(attIdx);

            if(att.isNominal() && attIdx != inputFormat.classIndex()) {
                newAtts.add(new Attribute(att.name() + "_mean_encoded"));
            }
            else {
                newAtts.add((Attribute)att.copy());
            }
        }

        Instances data = new Instances(inputFormat.relationName(), newAtts, 0);
        data.setClassIndex(inputFormat.classIndex());

        return data;
    }

    protected Instances process(Instances inputs) {

        Instances outputs = determineOutputFormat(inputs);

        if(m_Codes == null) {
            m_AttToBeModified = new boolean[inputs.numAttributes()];
            m_Codes = new double[inputs.numAttributes()][];

            double meanClass;
            AttributeStats classStats = inputs.attributeStats(inputs.classIndex());

            if(inputs.classAttribute().isNominal()) {
                meanClass = (double)classStats.nominalCounts[1] /
                    (double)(classStats.nominalCounts[0] + classStats.nominalCounts[1]);
            }
            else {
                meanClass = classStats.numericStats.mean;
            }

            for(int attIdx : m_SelectedAttributes) {
                Attribute att = inputs.attribute(attIdx);
    
                if(att.isNominal() && attIdx != inputs.classIndex()) {
                    m_Codes[attIdx] = new double[att.numValues()];
                    m_AttToBeModified[attIdx] = true;

                    for(int i = 0; i < inputs.numInstances(); i++) {
                        Instance inst = inputs.instance(i);
                        m_Codes[attIdx][(int)inst.value(attIdx)] += inst.classValue();
                    }

                   int[] counts = inputs.attributeStats(attIdx).nominalCounts;

                    for(int i = 0; i < m_Codes[attIdx].length; i++) {
                        if(counts[i] > 0) {
                            m_Codes[attIdx][i] /= (double)counts[i];
                        }
                        else {
                            m_Codes[attIdx][i] = meanClass;
                        }
                    }
                }
            }
        }

        for(int i = 0; i < inputs.numInstances(); i++) {
            Instance inst = inputs.instance(i);
            double[] vals = inst.toDoubleArray();

            for(int attIdx : m_SelectedAttributes) {
                if(m_AttToBeModified[attIdx]) {
                    vals[attIdx] = m_Codes[attIdx][(int)vals[attIdx]];
                }
            }

            outputs.add(new DenseInstance(inst.weight(), vals));
        }

        return outputs;
    }

    public static void main(String[] args) {
        runFilter(new NominalToMeanClass(), args);
    }
}
